
import sys
from hyperparameters import * 
import os
# Keep using keras-2 (tf-keras) rather than keras-3 (keras).
#os.environ['TF_USE_LEGACY_KERAS'] = '1'
import time
import numpy as np
import random 
import pandas as pd


#from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
#from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
#from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts


import reverb
import tensorflow as tf
from tensorflow import keras

#from tf_agents.agents.reinforce import reinforce_agent
#from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network

#from tf_agents.drivers import py_driver
#from tf_agents.environments import suite_pybullet

from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import policy_saver

#from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils

#from tf_agents.specs import tensor_spec
#from tf_agents.trajectories import trajectory
from tf_agents.utils import common


from tf_agents.metrics import py_metrics
from tf_agents.policies import random_py_policy
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils

#################################################################################################################################################################
def run_simulation(input_value, num_iterations, learning_path, saving_path):
	"""Starts a new environment and new agent and trains it. It takes an input value as a start time to train an agent in the 30min increment.

	The majority of the code was taken from Tensorflow SAC agent tutorial.

	Args:
		input_value (float): Which time of the day is the agent getting trained on. 

	Returns:
		No return: Instead saves checkpoint which is all of the saved information from a trained agent. 


	
	"""
	def generate_value_func_name(start_time):
		return f"value_function-{start_time}.txt"
	def generate_optimal_action_name(start_time):
		return f"optimal_actions-{start_time}.txt"

	# Set up fresh environment
	from environment_better import SolarEnvironment
	environment = SolarEnvironment()
	environment.update_start_time(input_value)

	collect_env = environment
	eval_env = environment

	

	use_gpu = True

	strategy = strategy_utils.get_strategy(tpu=False, use_gpu=False)

	observation_spec, action_spec, time_step_spec = (
		spec_utils.get_tensor_specs(collect_env))


	from agent_manager import actor_agent, critic_agent, tf_agent_init
	critic_net = critic_agent(collect_env, strategy)
	actor_net = actor_agent(collect_env, strategy)
	train_step, tf_agent = tf_agent_init(collect_env, actor_net, critic_net, strategy)

	#################################################################################################################################################################

	# Reverb code

	rate_limiter=reverb.rate_limiters.SampleToInsertRatio(samples_per_insert=3.0, min_size_to_sample=3, error_buffer=3.0)

	table_name = 'uniform_table'
	table = reverb.Table(
		table_name,
		max_size=replay_buffer_capacity,
		sampler=reverb.selectors.Uniform(),
		remover=reverb.selectors.Fifo(),
		rate_limiter=reverb.rate_limiters.MinSize(1))

	reverb_server = reverb.Server([table])


	# Since the SAC Agent needs both the current and next observation to compute the loss, we set sequence_length=2.
	reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
		tf_agent.collect_data_spec,
		sequence_length=2,
		table_name=table_name,
		local_server=reverb_server)

	# Now we generate a TensorFlow dataset from the Reverb replay buffer. We will pass this to the Learner to sample experiences for training.
	dataset = reverb_replay.as_dataset(
		sample_batch_size=batch_size, num_steps=2).prefetch(50)
	experience_dataset_fn = lambda: dataset

	#################################################################################################################################################################

	tf_eval_policy = tf_agent.policy
	eval_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_eval_policy, use_tf_function=True)

	tf_collect_policy = tf_agent.collect_policy
	collect_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_collect_policy, use_tf_function=True)

	random_policy = random_py_policy.RandomPyPolicy(collect_env.time_step_spec(), collect_env.action_spec())

	# We're storing trajectories for frames [(t0,t1) (t1,t2) (t2,t3), ...] because stride_length=1.
	rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
	reverb_replay.py_client,
	table_name,
	sequence_length=2,
	stride_length=1)



	# We create an Actor with the random policy and collect experiences to seed the replay buffer with.
	initial_collect_actor = actor.Actor(
	collect_env,
	random_policy,
	train_step,
	steps_per_run=100,
	observers=[rb_observer])
	initial_collect_actor.run()

	# Instantiate an Actor with the collect policy to gather more experiences during training.
	env_step_metric = py_metrics.EnvironmentSteps()
	collect_actor = actor.Actor(
	collect_env,
	collect_policy,
	train_step,
	steps_per_run=1,
	metrics=actor.collect_metrics(10),
	summary_dir=os.path.join(learning_path, learner.TRAIN_DIR),
	observers=[rb_observer, env_step_metric])

	# Create an Actor which will be used to evaluate the policy during training. We pass in actor.eval_metrics(num_eval_episodes) to log metrics later.
	eval_actor = actor.Actor(
	eval_env,
	eval_policy,
	train_step,
	episodes_per_run=num_eval_episodes,
	metrics=actor.eval_metrics(num_eval_episodes),
	summary_dir=os.path.join(learning_path, 'eval'),
	)

	saved_model_dir = os.path.join(learning_path, learner.POLICY_SAVED_MODEL_DIR)

	# Triggers to save the agent's policy checkpoints.
	learning_triggers = [
		triggers.PolicySavedModelTrigger(
			saved_model_dir,
			tf_agent,
			train_step,
			interval=policy_save_interval),
		triggers.StepPerSecondLogTrigger(train_step, interval=1000),
	]

	agent_learner = learner.Learner(
	learning_path,
	train_step,
	tf_agent,
	experience_dataset_fn,
	triggers=learning_triggers,
	strategy=strategy)


	def get_eval_metrics():
		eval_actor.run()
		results = {}
		for metric in eval_actor.metrics:
			results[metric.name] = metric.result()
		return results
	
	#metrics = get_eval_metrics()

	def log_eval_metrics(step, metrics, file):
		eval_results = (', ').join(
			'{} = {:.6f}'.format(name, result) for name, result in metrics.items())
		
		print((step, eval_results), file = file)
		

	# log_eval_metrics(0, metrics)
	
	######################################################################################################################################################################

	# Starting training here.



	def generate_checkpoint_name(start_time): 
		return f"checkpoint-{start_time}"	
	
	# Reset the train step
	tf_agent.train_step_counter.assign(0)

	# Evaluate the agent's policy once before training.
	#avg_return = get_eval_metrics()["AverageReturn"]
	#returns = [avg_return]
	returns = []
	log_path_eval = os.getenv("chosen_path_log_eval")
	log_file_eval = open(log_path_eval ,"a")
	
	log_path_loss = os.getenv("chosen_path_log_loss")
	log_file_loss = open(log_path_loss ,"a")



	# Saving agent into a checkpoint here. 

	base_path = saving_path
	name = generate_checkpoint_name(float(input_value))
	checkpoint_dir = f"{base_path}/{name}"

	# checkpoint = tf.train.Checkpoint(agent=tf_agent)
	# checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)

	# checkpoint_manager.save()

	# Setup the Checkpointer
	train_checkpointer = common.Checkpointer(
		ckpt_dir=checkpoint_dir,
		max_to_keep=1,
		agent=tf_agent,
		policy=tf_agent.policy,
		global_step=train_step
	)


	# reuse_weights = os.getenv("reuse_weights")
	# if int(reuse_weights) == 0:
	# 	pass
	# else:
	# 	if int(input_value) == 1410: 
	# 		pass
	# 	else:
	# 		restore_checkpoint()

	for _ in range(num_iterations):
		# Training.
		collect_actor.run()
		loss_info = agent_learner.run(iterations=1)

		# Evaluating.
		step = agent_learner.train_step_numpy

		# if eval_interval and step % eval_interval == 0:
		# 	metrics = get_eval_metrics()
		# 	log_eval_metrics(step, metrics, log_file_eval)
		# 	returns.append(metrics["AverageReturn"])

		if log_interval and step % log_interval == 0:
			print((step,loss_info.loss.numpy()), file=log_file_loss)

			print('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))
	
	log_file_loss.close()
	log_file_eval.close()

	train_checkpointer.save(train_step)

	# P_esm=float(os.getenv("P_esm"))  # Electrical storage charge limit  
	# P_esM=float(os.getenv("P_esM"))  # Electrical storage discharge limit (negative value indicates discharge)
	# T_esm=float(os.getenv("T_esm"))  # Thermal storage charge limit 
	# T_esM=float(os.getenv("T_esM"))  # Thermal storage discharge limit (negative value indicates discharge)

	# action_list_elec = np.linspace(P_esM, P_esm, 10) 
	# action_list_therm = np.linspace(T_esM, T_esm, 10)

	# # Create meshgrid
	# elec_grid, therm_grid = np.meshgrid(action_list_elec, action_list_therm)

	# # Stack into a single 2D array of combinations
	# combinations = np.stack((elec_grid, therm_grid), axis=-1)

	# # Reshape to (n_combinations, 2) where n_combinations = 10 * 10
	# actions_2d = combinations.reshape(-1, 2)

	


	# # Create an array for the first dimension with values from 20 to 100,
	# first_dim = np.linspace(20, 101, 81)

	# # Create an array for the second dimension with values from 0 to 100, repeated 81 times
	# second_dim = np.linspace(0, 101, 101)

	# # Create a meshgrid for combining these two into a 2D array
	# observations_2d = np.array(np.meshgrid(first_dim, second_dim))

	# values_dict = {}
	# for observation_list in observations_2d.T:
	# 	for observation in observation_list:
	# 		observation = np.append(observation, input_value/step_size)
	# 		observation = observation.reshape(1,3)
	# 		observation = tf.convert_to_tensor(observation, dtype = tf.float32)
	# 		q_a_list = []
	# 		for action in actions_2d:
	# 				action = action.reshape(2,)
	# 				action = np.expand_dims(action, axis=0)
	# 				action = tf.convert_to_tensor(action, dtype = tf.float32)
	# 				q_values, _ = tf_agent._target_critic_network_1((observation, action), training = True)
	# 				q_a_list.append([q_values[0], action[0]])

	# 		key_str = np.array2string(observation.numpy())

	# 		values_dict[key_str] = q_a_list
	
	# value_func_path = os.getenv("value_function_path")
	# os.makedirs(value_func_path, exist_ok=True)

	# name1 = generate_value_func_name(float(input_value))
	# value_func_path = f"{value_func_path}/{name1}"
	
	# import pprint
	# value_function_list = pprint.pformat(values_dict)
	# with open(value_func_path, "w") as file:
	# 		file.write(value_function_list)	



	rb_observer.close()
	reverb_server.stop()

	return input_value



if __name__ == "__main__":
	input_value = int(sys.argv[1])
	num_iterations = int(sys.argv[2])
	learning_path = sys.argv[3]
	saving_path = sys.argv[4]
	run_simulation(input_value, num_iterations, learning_path, saving_path)





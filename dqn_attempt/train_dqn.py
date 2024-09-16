from __future__ import absolute_import, division, print_function

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import base64
import sys
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import reverb

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.environments import wrappers
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.train.utils import train_utils

from hyperparameters_dqn import *

from dqn_manager import *

def run_dqn_train(input_value, num_iterations, saving_path):
	def generate_checkpoint_name(start_time): 
		return f"checkpoint-{start_time}"	
	def generate_value_func_name(start_time):
		return f"value_function-{start_time}.txt"
	def generate_optimal_action_name(start_time):
		return f"optimal_actions-{start_time}.txt"
	# Set up fresh environment
	from dqn_environment import SolarEnvironment
	environment = SolarEnvironment()
	environment.update_start_time(input_value)
 
	# train_py_env = wrappers.ActionDiscretizeWrapper(environment, num_actions=3)
	# eval_py_env = wrappers.ActionDiscretizeWrapper(environment, num_actions=3)
 
	train_py_env = environment
	eval_py_env = environment

	train_env = tf_py_environment.TFPyEnvironment(train_py_env)
	eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

	action_tensor_spec = tensor_spec.from_spec(train_py_env.action_spec())

	num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

	base_path = saving_path
	name = generate_checkpoint_name(float(input_value))
	checkpoint_dir = f"{base_path}/{name}"

	agent, _, global_step = start_dqn_agent(environment)
	agent.initialize()
	# # Define a helper function to create Dense layers configured with the right
	# # activation and kernel initializer.
	# def dense_layer(num_units):
	# 	return tf.keras.layers.Dense(
	# 	num_units,
	# 	activation=tf.keras.activations.relu,
	# 	kernel_initializer=tf.keras.initializers.VarianceScaling(
	# 		scale=2.0, mode='fan_in', distribution='truncated_normal'))

	# # QNetwork consists of a sequence of Dense layers followed by a dense layer
	# # with `num_actions` units to generate one q_value per available action as
	# # its output.
	# dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
	# q_values_layer = tf.keras.layers.Dense(
	# 	num_actions,
	# 	activation=None,
	# 	kernel_initializer=tf.keras.initializers.RandomUniform(
	# 		minval=-0.03, maxval=0.03),
	# 	bias_initializer=tf.keras.initializers.Constant(-0.2))
	# q_net = sequential.Sequential(dense_layers + [q_values_layer])
	
	# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

	# train_step_counter = tf.Variable(0)
	# global_step = tf.compat.v1.train.get_or_create_global_step()

	# agent = dqn_agent.DqnAgent(
	# 	train_env.time_step_spec(),
	# 	train_env.action_spec(),
	# 	q_network=q_net,
	# 	optimizer=optimizer,
	# 	td_errors_loss_fn=common.element_wise_squared_loss,
	# 	train_step_counter=train_step_counter)

	# agent.initialize()

	eval_policy = agent.policy
	collect_policy = agent.collect_policy
	random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())
	# Metrics and evaluation
	def compute_avg_return(environment, policy, num_episodes=10):

		total_return = 0.0
		for _ in range(num_episodes):

			time_step = environment.reset()
			episode_return = 0.0

			while not time_step.is_last():
				action_step = policy.action(time_step)
				time_step = environment.step(action_step.action)
				episode_return += time_step.reward
				total_return += episode_return

		avg_return = total_return / num_episodes
		return avg_return.numpy()[0]

	# Replay buffer
	table_name = 'uniform_table'
	replay_buffer_signature = tensor_spec.from_spec(
		agent.collect_data_spec)
	replay_buffer_signature = tensor_spec.add_outer_dim(
		replay_buffer_signature)

	table = reverb.Table(
		table_name,
		max_size=replay_buffer_max_length,
		sampler=reverb.selectors.Uniform(),
		remover=reverb.selectors.Fifo(),
		rate_limiter=reverb.rate_limiters.MinSize(1),
		signature=replay_buffer_signature)

	reverb_server = reverb.Server([table])

	replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
		agent.collect_data_spec,
		table_name=table_name,
		sequence_length=2,
		local_server=reverb_server)

	rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
	replay_buffer.py_client,
	table_name,
	sequence_length=2)
	
 
	# # Data collection
	# py_driver.PyDriver(
    # environment,
    # py_tf_eager_policy.PyTFEagerPolicy(
    #   random_policy, use_tf_function=True),
    # [rb_observer],
    # max_steps=initial_collect_steps).run(train_py_env.reset())

	# Dataset generates trajectories with shape [Bx2x...]
	dataset = replay_buffer.as_dataset(
		num_parallel_calls=3,
		sample_batch_size=batch_size,
		num_steps=2).prefetch(3)

	iterator = iter(dataset)





	
	log_path_eval = os.getenv("chosen_path_log_eval")
	log_file_eval = open(log_path_eval ,"a")
	
	log_path_loss = os.getenv("chosen_path_log_loss")
	log_file_loss = open(log_path_loss ,"a")

	# reuse_weights = os.getenv("reuse_weights")
	# if reuse_weights == 0:
	# 	pass
	# else:
	# 	if start_time == 1410: 
	# 		pass
	# 	else:
	# 		restore_name = generate_checkpoint_name(start_time)
	# 		base = saving_path
	# 		restore_dir = f"{base}/{restore_name}"
	# 		checkpoint = tf.train.Checkpoint(agent=agent)
	# 		checkpoint.restore(tf.train.latest_checkpoint(restore_dir))
	
 	# # Saving agent into a checkpoint here. 



	train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=global_step
)


	# (Optional) Optimize by wrapping some of the code in a graph using TF function.
	agent.train = common.function(agent.train)

	# Reset the train step.
	agent.train_step_counter.assign(0)

	# Evaluate the agent's policy once before training.
	avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
	returns = [avg_return]

	# Reset the environment.
	time_step = train_py_env.reset()

	# Create a driver to collect experience.
	collect_driver = py_driver.PyDriver(
		environment,
		py_tf_eager_policy.PyTFEagerPolicy(
		agent.collect_policy, use_tf_function=True),
		[rb_observer],
		max_steps=collect_steps_per_iteration)

	for _ in range(num_iterations):
		
		# Collect a few steps and save to the replay buffer.
		time_step, _ = collect_driver.run(time_step)

		# Sample a batch of data from the buffer and update the agent's network.
		experience, unused_info = next(iterator)
		train_loss = agent.train(experience).loss

		step = agent.train_step_counter.numpy()

		if step % log_interval == 0:
			print('step = {0}: loss = {1}'.format(step, train_loss))
			print((step, train_loss), file=log_file_loss)

		if step % eval_interval == 0:
			avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
			print('step = {0}: Average Return = {1}'.format(step, avg_return))
			print((step, avg_return), file=log_file_eval)

			returns.append(avg_return)

	log_file_loss.close()
	log_file_eval.close()

	train_checkpointer.save(global_step)
	
	def get_value_function(agent, state):
		"""Retrieves the q network from the agent and finds the max Q(s,a) and uses it as a value function

		Args:
			agent (TF_agent): a trained agent
			state (tensor): the state of the environment

		Returns:
			float: V(s)
			list: Q(s,a) for all a
		"""
		# Convert state to a batch format (required by the network)

		# state = tf.convert_to_tensor(state)
		# state = tf.expand_dims(state, axis=0)  # Add batch dimension
		# Compute Q-values using the Q-network

		q_values, _ = agent._q_network(state.numpy())  # Ignore the network state output

		# Extract the maximum Q-value (i.e., the value of the state)
		value_function = q_values[0].numpy().max() # Maximum Q-value across actions
		# If the agent behaved optimally henceforth it would have this value.
		return value_function, q_values[0].numpy()

	value_function_list = []
	optimal_action_list = []
	state = np.array([0, (input_value)/step_size])
	for i in range (0,101):
		# Create a NumPy array with 2 elements
		state[0] = int(i)

		# Reshape the NumPy array to the desired shape (1, 2)
		numpy_state = state.reshape(1, 2)

		# Convert the reshaped NumPy array to a TensorFlow tensor with dtype float32
		tf_tensor_state = tf.convert_to_tensor(numpy_state, dtype=tf.float32)
		
		value, action_values= get_value_function(agent, tf_tensor_state)

		value_function_list.append(value)
		optimal_action_list.append(np.argmax(action_values)) # Retrieves the index of the best action. 

	# Saving the value function
	value_func_path = os.getenv("value_function_path")
	os.makedirs(value_func_path, exist_ok=True)

	name1 = generate_value_func_name(float(input_value))
	value_func_path = f"{value_func_path}/{name1}"
	
	with open(value_func_path, "w") as file:
		for item in value_function_list:
			file.write(str(item) + "\n")

	# Saving the optimal policy
	optimal_action_path = os.getenv("optimal_action_path")
	name2 = generate_optimal_action_name(float(input_value))
	os.makedirs(optimal_action_path, exist_ok=True)
	optimal_action_path = f"{optimal_action_path}/{name2}"
	
	with open(optimal_action_path, "w") as file:
		for item in optimal_action_list:
			file.write(str(item) + "\n")
	
	rb_observer.close()
	reverb_server.stop()

	
	return None

if __name__ == "__main__":
	input_value = int(sys.argv[1])
	num_iterations = int(sys.argv[2])
	saving_path = sys.argv[3]
	run_dqn_train(input_value, num_iterations, saving_path)
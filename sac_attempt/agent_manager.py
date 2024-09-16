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
from tf_agents.agents.sac.sac_agent import SacAgent


strategy = strategy_utils.get_strategy(tpu=False, use_gpu=False)

def critic_agent(collect_env, strategy):
	# Creating the critic agent
	observation_spec, action_spec, time_step_spec = (spec_utils.get_tensor_specs(collect_env))

	with strategy.scope():
		critic_net = critic_network.CriticNetwork(
		(observation_spec, action_spec),
		observation_fc_layer_params=None,
		action_fc_layer_params=None,
		joint_fc_layer_params=critic_joint_fc_layer_params,
		kernel_initializer='glorot_uniform',
		last_kernel_initializer='glorot_uniform')

	return critic_net 

def actor_agent(collect_env, strategy):
	observation_spec, action_spec, time_step_spec = (spec_utils.get_tensor_specs(collect_env))

	with strategy.scope():
		actor_net = actor_distribution_network.ActorDistributionNetwork(
		observation_spec,
		action_spec,
		fc_layer_params=actor_fc_layer_params,
		continuous_projection_net=(
		tanh_normal_projection_network.TanhNormalProjectionNetwork))
	return actor_net

def tf_agent_init(collect_env, actor_net, critic_net, strategy):
	observation_spec, action_spec, time_step_spec = (spec_utils.get_tensor_specs(collect_env))
	train_step = train_utils.create_train_step()
	with strategy.scope():
		
		tf_agent = sac_agent.SacAgent(
		time_step_spec,
		action_spec,
		actor_network=actor_net,
		critic_network=critic_net,
		actor_optimizer=tf.keras.optimizers.Adam(
			learning_rate=actor_learning_rate),
		critic_optimizer=tf.keras.optimizers.Adam(
			learning_rate=critic_learning_rate),
		alpha_optimizer=tf.keras.optimizers.Adam(
			learning_rate=alpha_learning_rate),
		target_update_tau=target_update_tau,
		target_update_period=target_update_period,
		td_errors_loss_fn=tf.math.squared_difference,
		gamma=gamma,
		reward_scale_factor=reward_scale_factor,
		train_step_counter=train_step)
		tf_agent.initialize()


	return train_step, tf_agent

def compute_value_function(observation, environment, tf_agent):
	tf_env = tf_py_environment.TFPyEnvironment(environment)

	# Get the time_step_spec from the policy to create a matching TimeStep
	time_step_spec = tf_agent.policy.time_step_spec

	# Create a dummy TimeStep with the correct structure
	dummy_time_step = tf_env.reset()

	# Replace the observation in the dummy TimeStep with the actual observation
	time_step = ts.TimeStep(
		step_type=dummy_time_step.step_type,
		reward=dummy_time_step.reward,
		discount=dummy_time_step.discount,
		observation=observation
	)

	# Get action distribution from the policy
	policy = tf_agent.policy
	action_step = policy.action(time_step)

	# Sample an action
	action = action_step.action
	_, log_prob = tf_agent._actions_and_log_probs(dummy_time_step, training=True)

	# Get Q-values from the critic network(s)
	pred_input = (observation[0], action)
	q_values, _ = tf_agent._critic_network_1(
		  pred_input, step_type=dummy_time_step.step_type, training=True)
	min_q_value = tf.reduce_min(q_values, axis=0)

	# Compute the value function
	alpha = float(os.getenv('alpha_learning_rate'))
	value_function = min_q_value - alpha * log_prob

	#TODO: Test it with different scenarios to check if logic is sound, 90 or 100 etc etc
	return value_function

# Example usage
# observation = tf.constant(tf_env.reset().observation)
# value_function = compute_value_function(observation, tf_agent)
# print("Value function:", value_function.numpy())
def reverb_table(tf_agent):
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

class AgentEnsemble:
	def __init__(self, environment, checkpoint_directory):
		
		self.environment = environment
		self._agent_map = {}
		for folder_name in os.listdir(checkpoint_directory):
			full_path = f"{checkpoint_directory}/{folder_name}"
			if os.path.isdir(full_path):
				agent_start_time = self.get_start_time_from_checkpoint_name(folder_name)
				self._agent_map[agent_start_time] = self._load_agent_from_checkpoint(full_path)

	def get_start_time_from_checkpoint_name(self, folder_dir):

		return float(folder_dir.split("-")[1])

	def _load_agent_from_checkpoint(self, checkpoint_dir):
		critic_net = critic_agent(self.environment, strategy)
		actor_net = actor_agent(self.environment, strategy)
		train_step, tf_agent = tf_agent_init(self.environment, actor_net, critic_net, strategy)


		# Setup the Checkpointer
		train_checkpointer = common.Checkpointer(
			ckpt_dir=checkpoint_dir,
			max_to_keep=1,
			agent=tf_agent,
			policy=tf_agent.policy,
			global_step=train_step
		)

		# Function to save a checkpoint
		def save_checkpoint():
			train_checkpointer.save(global_step=train_step)

		# Function to restore from the latest checkpoint
		def restore_checkpoint():
			train_checkpointer.initialize_or_restore()
		
		restore_checkpoint()
		return tf_agent

	def __getitem__(self, start_time) -> SacAgent:
		return self._agent_map[start_time]
	
	def __setitem__(self, start_time: int, agent: SacAgent):
		self._agent_map[start_time] = agent
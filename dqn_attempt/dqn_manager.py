from __future__ import absolute_import, division, print_function

import os

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

def generate_checkpoint_name(start_time):
    return f"checkpoint-{start_time}"
def start_dqn_agent(environment):

	train_py_env = environment
	eval_py_env = environment

	train_env = tf_py_environment.TFPyEnvironment(train_py_env)
	eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

	action_tensor_spec = tensor_spec.from_spec(train_py_env.action_spec())
	num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
 
    # Define a helper function to create Dense layers configured with the right
	# activation and kernel initializer.
	def dense_layer(num_units):
		return tf.keras.layers.Dense(
		num_units,
		activation=tf.keras.activations.relu,
		kernel_initializer=tf.keras.initializers.VarianceScaling(
			scale=2.0, mode='fan_in', distribution='truncated_normal'))

	# QNetwork consists of a sequence of Dense layers followed by a dense layer
	# with `num_actions` units to generate one q_value per available action as
	# its output.
	dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
	q_values_layer = tf.keras.layers.Dense(
		num_actions,
		activation=None,
		kernel_initializer=tf.keras.initializers.RandomUniform(
			minval=-0.03, maxval=0.03),
		bias_initializer=tf.keras.initializers.Constant(-0.2))
	q_net = sequential.Sequential(dense_layers + [q_values_layer])
	
	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

	train_step_counter = tf.Variable(0)
	global_step = tf.compat.v1.train.get_or_create_global_step()

	agent = dqn_agent.DqnAgent(
		train_env.time_step_spec(),
		train_env.action_spec(),
		q_network=q_net,
		optimizer=optimizer,
		td_errors_loss_fn=common.element_wise_squared_loss,
		train_step_counter=train_step_counter)

	#agent.initialize()

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
	
 
	# Data collection
	py_driver.PyDriver(
    environment,
    py_tf_eager_policy.PyTFEagerPolicy(
      random_policy, use_tf_function=True),
    [rb_observer],
    max_steps=initial_collect_steps).run(train_py_env.reset())

	# Dataset generates trajectories with shape [Bx2x...]
	dataset = replay_buffer.as_dataset(
		num_parallel_calls=3,
		sample_batch_size=batch_size,
		num_steps=2).prefetch(3)

	iterator = iter(dataset)
	return agent, replay_buffer, global_step
	
    # base_path = saving_path
	# name = generate_checkpoint_name(float(input_value))
	# checkpoint_dir = f"{base_path}/{name}"
def load_dqn_agent(checkpoint_dir, agent, replay_buffer, global_step):
    train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=global_step)
    return train_checkpointer
# Function to extract the value function for a given state
def get_value_function(agent, state):
    # Convert state to a batch format (required by the network)
    
    state = tf.convert_to_tensor(state)
    state = tf.expand_dims(state, axis=0)  # Add batch dimension
	
    # Compute Q-values using the Q-network
    q_values, _ = agent._q_network(state)  # Ignore the network state output

    # Extract the maximum Q-value (i.e., the value of the state)
    value_function = q_values[0].numpy().max() # Maximum Q-value across actions
    # If the agent behaved optimally henceforth it would have this value.
    return value_function, q_values[0].numpy()
def get_value_function_list(agent, state):
	value_function_list = []
	for i in range (20,101):
		# Create a NumPy array with 2 elements
		state[0] = int(i)

		# Reshape the NumPy array to the desired shape (1, 2)
		numpy_state = state.reshape(1, 2)

		# Convert the reshaped NumPy array to a TensorFlow tensor with dtype float32
		tf_tensor_state = tf.convert_to_tensor(numpy_state, dtype=tf.float32)
		
		value = get_value_function(agent, tf_tensor_state)
		value_function_list.append(value)
	return value_function_list
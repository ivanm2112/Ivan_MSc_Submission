# %%


import os
from dotenv import load_dotenv
load_dotenv("./envs_sac/default.env")
# Keep using keras-2 (tf-keras) rather than keras-3 (keras).
#os.environ['TF_USE_LEGACY_KERAS'] = '1'
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
from tf_agents.policies import greedy_policy
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils
from tf_agents.agents.sac.sac_agent import SacAgent

from hyperparameters import *

# %%
from agent_manager import critic_agent, actor_agent, tf_agent_init
# Set up fresh environment


def setup_environment_agent(environment, time):
    collect_env = environment

    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=False)

    critic_net = critic_agent(collect_env, strategy)
    actor_net = actor_agent(collect_env, strategy)
    train_step, tf_agent2 = tf_agent_init(collect_env, actor_net, critic_net, strategy)
    
    base_path = os.getenv("chosen_path_saving")

    name = generate_checkpoint_name(float(time))

    checkpoint_dir = f"{base_path}/{name}"

    train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=tf_agent2,
    policy=tf_agent2.policy,
    global_step=train_step
    )

    train_checkpointer.initialize_or_restore().expect_partial()

    return tf_agent2

def generate_checkpoint_name(start_time):
    return f"checkpoint-{start_time}"

def get_value_function_value(environment, tf_agent):
    """
    time is the time after +step_size of current time
    """

    tf_env = tf_py_environment.TFPyEnvironment(environment)
    time_step = tf_env._current_time_step()
    
    observation = time_step.observation
    policy = tf_agent.policy
    #deterministic_policy = greedy_policy.GreedyPolicy(policy)

    # Extract the action from the action_step
    #action_step = deterministic_policy.action(time_step)

    action_distribution = policy.distribution(time_step).action
    action = action_distribution.mean() # Convert to numpy array
    action2 = np.expand_dims(action, axis=0)

    q_values, _ = tf_agent._target_critic_network_1((observation, action2), training = True)

    _, log_prob = tf_agent._actions_and_log_probs(time_step, training=True)

    # Get Q-values from the critic network(s)
    min_q_value = tf.reduce_min(q_values, axis=0)

    # Compute the value function
    alpha = float(os.getenv('alpha_learning_rate'))
    value_function = min_q_value - alpha * log_prob
    #print(value_function[0].numpy())

    return value_function[0].numpy()

# # print('Previous state:', time_step.observation)
# # print('Action taken:', action_step.action)
# # print('Reward received:', next_time_step.reward)
# # print('Next state:', next_time_step.observation)
# # %%

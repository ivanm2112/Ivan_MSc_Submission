# Imports
import os
# Keep using keras-2 (tf-keras) rather than keras-3 (keras).
#os.environ['TF_USE_LEGACY_KERAS'] = '1'
import time
import math
from dotenv import load_dotenv
import numpy as np
import random 
from random import randint
import pandas as pd
from copy import deepcopy

#load_dotenv("./envs/simple1.env")


import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment 
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts



from tf_agents.metrics import py_metrics
from tf_agents.policies import random_py_policy
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils

#########################################################################################################################################################################################################
from hyperparameters_dqn import *
which_forecast = int(os.getenv("which_forecast"))
if which_forecast==1:
    from forecasts_simple1 import *
elif which_forecast==2: 
    from forecasts_simple2 import *
# elif which_forecast==3: 
#     from forecasts_simple3 import *
# elif which_forecast==4: 
#     from forecasts_simple4 import *
else:
    from forecasts import *
#########################################################################################################################################################################################################
C_esM=float(os.getenv("C_esM"))  # Electrical storage capacity
C_esm=float(os.getenv("C_esm"))  # Electrical storage minimum charge 

C_tsM=float(os.getenv("C_tsM"))  # Thermal storage capacity 
C_tsm=float(os.getenv("C_tsm"))  # Thermal storage minimum charge 

P_esm=float(os.getenv("P_esm"))  # Electrical storage charge limit  
P_esM=float(os.getenv("P_esM"))  # Electrical storage discharge limit (negative value indicates discharge)
T_esm=float(os.getenv("T_esm"))  # Thermal storage charge limit 
T_esM=float(os.getenv("T_esM"))  # Thermal storage discharge limit (negative value indicates discharge)

eta_WH=float(os.getenv("eta_WH"))  # Water heater converter efficiency 
eta_PV=float(os.getenv("eta_PV"))  # PV array converter efficiency
eta_C_es=float(os.getenv("eta_C_es"))  # Electrical storage charge efficiency 
eta_D_es=float(os.getenv("eta_D_es"))  # Electrical storage discharge efficiency 
eta_C_ts=float(os.getenv("eta_C_ts"))  # Thermal storage charge efficiency 
eta_D_ts=float(os.getenv("eta_D_ts"))  # Thermal storage discharge efficiency 

D_es=float(os.getenv("D_es"))  # Electrical storage self-discharge rate 
D_ts=float(os.getenv("D_ts"))  # Thermal storage self-discharge rate 

D_pes=float(os.getenv("D_pes"))  # Electrical storage self-discharge rate as a percentage
D_pts=float(os.getenv("D_pts"))  # Thermal storage self-discharge rate as a percentage

F_SD=float(os.getenv("F_SD"))  # Cost of battery degradation
#########################################################################################################################################################################################################

# D_pes = (D_es/60)/(1000*C_esM)        # Electrical storage self-discharge rate as a percent but not in percentage form ie. 0.07 instead of 7%
# D_pts = (D_ts/60)/(1000*C_tsM)        # Thermal storage self-discharge rate as a percent but not in percentage form ie. 0.07 instead of 7%

# F_SD = 10 ** (-6)      # Cost of battery degradation, selected empirically, taken from Challen_et_al
#########################################################################################################################################################################################################

def turn_battery_to_int(number: float) -> int:
    # Truncate the decimal part of the number
    truncated = math.trunc(number)
    decimal_part = number - truncated
    
    # Determine the probabilities
    ceiling_probability = decimal_part
    
    # Randomly decide whether to return floor or ceiling based on the probabilities
    if random.random() < ceiling_probability:
        return math.ceil(number)
    else:
        return math.floor(number)
  
def generate_checkpoint_name(start_time): 
    return f"checkpoint-{start_time}"
def generate_value_func_name(start_time):
	return f"value_function-{start_time}.txt"
#from dqn_manager import load_dqn_agent, start_dqn_agent, get_value_function, get_value_function_list

# The start time thing is being funky af
#########################################################################################################################################################################################################    
class SolarEnvironment(py_environment.PyEnvironment):
    # First standard functions of a TensorFlow environment
    def __init__(self, start_time=start_time):
        self._start_time = start_time
        self._reset()

        self._action_spec = array_spec.BoundedArraySpec(  
        shape=(), dtype=np.int32, minimum=0, maximum=4, name='action') 

		# Remember to rescale
        self._action_scale = [P_esM, P_esM/2, 0, P_esm/2, P_esm]

        self._observation_spec = array_spec.BoundedArraySpec(
        shape=(2,), dtype=np.float32, minimum=[20.0, 0.0], name='observation') #now a float32, because sac needed it

        self._episode_ended = False

    
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec
    

    def get_start_time(self):
        return (self._state[1]-1)*30  #TODO: THIS IS BUGGY, NOT USED CURRENTLY
    
    def update_start_time(self, val):
        self._start_time = val  
        self._state[1] = val/step_size# The state is updated through reset at the start of each new train, or IS IT??

    # def get_value_function_from_time(self):
    #     if self._start_time != 1410:
    #         saving_path = os.getenv("chosen_path_saving")
    #         input_value = self._state[1]
    #         base_path = saving_path
    #         name = generate_checkpoint_name(float(input_value*30))
    #         checkpoint_dir = f"{base_path}/{name}"
            
    #         agent2, replay_buffer2, global_step2 = start_dqn_agent(self)
    #         train_checkpointer2 = load_dqn_agent(checkpoint_dir, agent2, replay_buffer2, global_step2)
            
    #         train_checkpointer2.initialize_or_restore()


    #         value_function_list = get_value_function_list(agent2, self._state)
    #     return value_function_list   
    def _reset(self):
        # First is electric, second is time

        self._state = [float(int(random.uniform(20.0, 100.0))), self._start_time/step_size]     # randomly initialize
        #self._state = [50,47] 
        
        # 

        self._episode_ended = False



        return ts.restart(np.array(self._state, dtype=np.float32))
    # Set coefficient to a new variable, so that it can load in from .env file


    ###############################################################################################################################################
    # Second to replicate equation (16) in the Challen_et_al paper, to choose whether to use charging or discharging efficiency

    def which_eta_electric(self, action):
        if action <= 0:
            eta_ES = eta_D_es
        if action > 0:
            eta_ES = eta_C_es
        return eta_ES
    
    #################################################################################################################################################################
    # Thirdly to replicate equation (21) from Challen_et_al, the change in battery percentages based on chosen (dis)charging amount

    def charge_discharge_electric(self, action, previous_electric):
        """
        Input: action[0], how much  the agent decided to charge the electric battery, positive is charge, negative is discharge

        Output: change in electric battery power in kWh.
        """
        # Put a coefficient here
        eta = self.which_eta_electric(self._action_scale[action])
        dCe =  (eta * self._action_scale[action])/(60/step_size)  - (D_pes*previous_electric)*(C_esM/100)/(60/step_size) #efficiency of (dis)charging times amount of charge minus passive loss
        
        return dCe # in kWh

    
    #################################################################################################################################################################
    # Cost per time step
    def running_cost(self, action, previous_electric):
        """The cost functional was taken from Challen_et_al (18). 
        
        Inputs: self._observation_spec[2] is the time of the system, self._action_spec[0] and self._action_spec[1] are the actions taken by the agent, for electric and thermal respectively

        # self._action_spec[0] and self._action_spec[1] are multiplied by their respective efficiencies which are given by the functions
    
        Returns:
            float: amount of pounds earned/lost

        """
        eta_electric = self.which_eta_electric(action)
        
        # The minus one in the index is added because the cost is calculated after time is updated.
        cost = (-Elec_Demand_Forecast[int(self._state[1])-1] + PV_forecast[int(self._state[1])-1])*(step_size/60) -\
                (eta_electric * self._action_scale[action])/(60/step_size)
        #TODO check signs

        if cost < 0:
            cost = cost * Elec_Buy[int(self._state[1])-1]
        else:
            cost = cost * Elec_Sell[int(self._state[1])-1]

        # Now adding the passive cost of batteries degrading
        electric_charge_percent = 100*((self.charge_discharge_electric(action, previous_electric))/C_esM)

        cost = cost - (electric_charge_percent ** 2) * F_SD  # This matches (19)

        return cost
    
    #################################################################################################################################################################
    # For future potential GUI
    def render(self):
        print(self._state)

    def get_state(self):
        return deepcopy(self._state)
    
    #################################################################################################################################################################
    # Standard step notation in TensorFlow 
    # TODO change your action changing into a discrete compatible thing
    def _step(self, action):
        
        # loads the value function list from the file
        # based on the time, if current time is 1380, then loads 1410
        value_function_value = 0 
 
        

        if self._episode_ended:
        # The last action ended the episode. Ignore the current action and start
        # a new episode.
            return self.reset()

        previous_electric = self._state[0]
        
        

        electric_charge_percent = 100*((self.charge_discharge_electric(action, previous_electric))/C_esM)

        


        # Accounting for precision of battery charge level. Created a function called turn_battery_to_int which randomly floors or ceilings the number 
        # and the probability of that happening is based on the decimal value of the number.
        # Also did not allow charge to get below minimum or above maximum.
        if electric_charge_percent < 0:
            if self._state[0]+electric_charge_percent < 0:
                action = 1  # Meaning take the 0th action
            else:
                self._state[0] = self._state[0]+electric_charge_percent

        else:
            if self._state[0]+electric_charge_percent > 100:  # Checks if the action would charge the battery above 100%
                action = 1               

            else:
                self._state[0] = self._state[0]+electric_charge_percent
        


        if self._start_time == 1410:
            value_function_value = 0    #Initial agent has no one to look up to
        else:
            value_func_path = os.getenv("value_function_path")
            input_value = self._start_time + 30
            name = generate_value_func_name(float(input_value))
            value_func_path = f"{value_func_path}/{name}"
            # Open the file in read mode
            with open(value_func_path, "r") as file:
                # Read each line and strip newline characters
                value_function_list = [line.strip() for line in file]

            value_function_value = float(value_function_list[int(self._state[0])])  # Batter at 0 gives 0th entry in list so its okay.
        
        self._state[1] = self._state[1] + 1
        if self._state[1] == int(end_time/step_size):

            self._episode_ended = True
        

        if self._episode_ended: return ts.termination(np.array(self._state, dtype=np.float32), 
            reward = self.running_cost(action, previous_electric) +\
                 (self._state[0])*C_esM/100*Elec_Sell[int(self._state[1])] ) # Matching (20)
        
        # print(self.running_cost(action, previous_electric)+value_function_value)
        # print(value_function_value)
        #reward = self.running_cost(action) # For testing purposes
        return ts.termination(np.array(self._state, dtype=np.float32), reward = self.running_cost(action, previous_electric) + value_function_value) 
        # not necessary


environment = SolarEnvironment()        
# Discretizing action space
# Wrapping in a tenssorflow environment

#tf_discrete_env = tf_py_environment.TFPyEnvironment(discrete_action_env)
tf_env = tf_py_environment.TFPyEnvironment(environment)

# print(isinstance(tf_discrete_env, tf_environment.TFEnvironment))
# print("TimeStep Specs:", tf_discrete_env.time_step_spec())
# print("Action Specs:", tf_discrete_env.action_spec())

# Setting environments
# train_py_env = discrete_action_env
# eval_py_env = discrete_action_env

# collect_env = suite_pybullet.load(env_name)
# eval_env = suite_pybullet.load(env_name)

collect_env = environment
eval_env = environment  

# reward = 0   # For testing purposes
# strategy = strategy_utils.get_strategy(tpu=False, use_gpu=False)

# from agent_manager import compute_value_function, AgentEnsemble

# agent_path = os.getenv("chosen_path_saving")
# agent_ens = AgentEnsemble(environment, agent_path)
#TODO change back to 20% min

#########################################################################################################################################################################################################
if __name__ =='__main__':
    environment = SolarEnvironment()
    
    print(utils.validate_py_environment(environment, episodes=5))

    load_dotenv("./envs/simple1.env")


    time_step = environment.reset()
    print(time_step)

    cumulative_reward = time_step.reward

    for i in range(47, 48):
        charge = [2, 0]
        charge_action = np.array(charge, dtype=np.float64)
        time_step = environment._step(charge_action)
        # if i in (21,22,23,24,25,26):
        #     print(time_step.reward)
        #     print(Elec_Demand_Forecast[i-1])
        #     print(Therm_forecast[i-1])
        #     print(PV_forecast[i-1])
        # if time_step.reward > 0:
        #     print(time_step.reward)
        reward = time_step.reward
        print(environment._state)
        print(reward)
        cumulative_reward += time_step.reward
        i +=1

    print('Final Reward = ', cumulative_reward)
    """
    agent = RandomAgent()
    episode_count = 1000
    step_count = 48
    for episode in range(episode_count):
        environment.reset()
        # This block of code represents one episode, can be extracted to a function, helpful prints could also be added
        for step in range(step_count):
            action = agent.pick(environment.action_space)
            new_state, reward = environment._step(action)
            total_cost += reward
            environment.render()
            print(total_cost)
            #time.sleep(1)
    """






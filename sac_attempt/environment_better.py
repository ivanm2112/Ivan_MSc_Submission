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

#load_dotenv("./envs_sac/simple_sac.env")


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
from test_sac_better import *
#########################################################################################################################################################################################################
from hyperparameters import *
which_forecast = int(os.getenv("which_forecast"))
if which_forecast==1:
    from forecasts_simple1 import *
# elif which_forecast==2: 
#     from forecasts_simple2 import *
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
  

    

#########################################################################################################################################################################################################    
class SolarEnvironment(py_environment.PyEnvironment):
    # First standard functions of a TensorFlow environment
    def __init__(self, start_time = start_time):
        self._start_time = start_time
        self._reset()

        self._action_spec = array_spec.BoundedArraySpec(  
        shape=(2,), dtype=np.float32, minimum=[P_esM, T_esM], maximum=[P_esm, T_esm], name='action')   


        self._observation_spec = array_spec.BoundedArraySpec(
        shape=(1,3), dtype=np.float32, minimum=[20.0, 0.0, 0.0], name='observation') #now a float32, because sac needed it

        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec
    

    def get_start_time(self):
        return (self._state[2]-1)*30  #TODO: THIS IS BUGGY, NOT USED CURRENTLY
    
    def update_start_time(self, val):
        self._start_time = val
        self._state[2] = val/step_size
        if self._start_time != 1410:
            self._environment_agent = setup_environment_agent(self, self._start_time+30)
        else:
            self._environment_agent = None
    
    def _reset(self):
        # First is electric, second is thermal battery charge level, third is time

        self._state = [float(int(random.uniform(20.0, 100.0))), float(int(random.uniform(0.0, 100.0))), self._start_time/step_size]     # randomly initialize
        #self._state = [50,50,0] 
        

        self._episode_ended = False



        return ts.restart(np.array([self._state], dtype=np.float32))

    


    ###############################################################################################################################################
    # Second to replicate equation (16) in the Challen_et_al paper, to choose whether to use charging or discharging efficiency

    def which_eta_electric(self, action):
        if action[0] <= 0:
            eta_ES = eta_D_es
        if action[0] > 0:
            eta_ES = eta_C_es
        return eta_ES
    

    def which_eta_thermal(self, action):
        if action[1] <= 0:
            eta_TS = eta_D_ts
        if action[1] > 0:
            eta_TS = eta_C_ts
        return eta_TS
    
    #################################################################################################################################################################
    # Thirdly to replicate equation (21) from Challen_et_al, the change in battery percentages based on chosen (dis)charging amount

    def charge_discharge_electric(self, action, previous_electric):
        """
        Input: action[0], how much  the agent decided to charge the electric battery, positive is charge, negative is discharge

        Output: change in electric battery power in kWh.
        """
        eta = self.which_eta_electric(action)
        dCe =  (eta * action[0])/(60/step_size)  - (D_pes*previous_electric)*(C_esM/100)/(60/step_size) #efficiency of (dis)charging times amount of charge minus passive loss
        
        return dCe


    def charge_discharge_thermal(self, action, previous_thermal):
        """
        Input: action[1], how much  the agent decided to charge the thermal battery, positive is charge, negative is discharge

        Output: change in thermal battery power in kWh.
        """
        eta = self.which_eta_thermal(action)
        dCh = (eta * action[1])/(60/step_size)  - (D_pts*previous_thermal)*(C_tsM/100)/(60/step_size) 
        
        return dCh
    
    #################################################################################################################################################################
    # Cost per time step
    def running_cost(self, action, previous_electric, previous_thermal):
        """The cost functional was taken from Challen_et_al (18). 
        
        Inputs: self._observation_spec[2] is thet time of the system, self._action_spec[0] and self._action_spec[1] are the actions taken by the agent, for electric and thermal respectively

        # self._action_spec[0] and self._action_spec[1] are multiplied by their respective efficiencies which are given by the functions
    
        Returns:
            float: amount of pounds earned/lost

        """
        eta_electric = self.which_eta_electric(action)
        eta_thermal = self.which_eta_thermal(action)
        
        # The minus one in the index is added because the cost is calculated after time is updated.
        cost = (-Elec_Demand_Forecast[int(self._state[2])-1] + PV_forecast[int(self._state[2])-1])*(step_size/60) -\
                (eta_electric * action[0])/(60/step_size) + \
                     min((-Therm_forecast[int(self._state[2])-1]*(step_size/60) - (eta_thermal * action[1])/(60/step_size))/eta_WH, 0)  

        #TODO check signs

        if cost < 0:
            cost = cost * Elec_Buy[int(self._state[2])-1]
        else:
            cost = cost * Elec_Sell[int(self._state[2])-1]

        # Now adding the passive cost of batteries degrading
        electric_charge_percent = 100*((self.charge_discharge_electric(action, previous_electric))/C_esM)
        thermal_charge_percent = 100*((self.charge_discharge_thermal(action, previous_thermal))/C_tsM)

        cost = cost - (electric_charge_percent ** 2 + thermal_charge_percent ** 2) * F_SD  # This matches (19)

        return cost
    
    #################################################################################################################################################################
    # For future potential GUI
    def render(self):
        print(self._state)

    def get_state(self):
        return deepcopy(self._state)
    
    #################################################################################################################################################################
    # Standard step notation in TensorFlow 

    def _step(self, action):
        value_function_value = 0
        self._state[2] = float(self._start_time/step_size)
        # if int(self._start_time) == 1410:
        #     value_function_value = 0    #Initial agent has no one to look up to
        # else:
        #     observation = [[[self._state[0],self._state[1],self._state[2]]]]
        #     observation = tf.constant(observation, dtype=tf.float32)
        #     value_function_value = compute_value_function(observation, environment, agent_ens[self._start_time + 30])
        

        if self._episode_ended:
        # The last action ended the episode. Ignore the current action and start
        # a new episode.
            return self.reset()

        previous_electric = self._state[0]
        previous_thermal = self._state[1]
        
        

        electric_charge_percent = 100*((self.charge_discharge_electric(action, previous_electric))/C_esM)
        thermal_charge_percent = 100*((self.charge_discharge_thermal(action, previous_thermal))/C_tsM)  #100 because it is a percentage, divided by ChM to see ratio charged

        


        # Accounting for precision of battery charge level. Created a function called turn_battery_to_int which randomly floors or ceilings the number 
        # and the probability of that happening is based on the decimal value of the number.
        # Also did not allow charge to get below minimum or above maximum. 
        if electric_charge_percent < 0:
            if self._state[0]+electric_charge_percent < 20:    # Check if the action would discharge the battery below the permitted level
                action[0] = (20 - self._state[0])*C_esM/100*(60/step_size)   #Change the action so that it only discharges the right amount
                self._state[0] = 20
            else:

                self._state[0] = turn_battery_to_int(self._state[0]+electric_charge_percent) 

        else:
            if self._state[0]+electric_charge_percent > 100:  # Checks if the action would charge the battery above 100%

                action[0] = (100 - self._state[0])*C_esM/100*(60/step_size)        # Reduces such action to the highest possible to reach 100%, (percentage * battery * time charging)
                self._state[0] = 100                

            else:
                self._state[0] = turn_battery_to_int(self._state[0]+electric_charge_percent)
        

        if thermal_charge_percent < 0:
            original_thermal_soc_store = self._state[1]
            if self._state[1]+thermal_charge_percent < 0:   # Check if the action would discharge the battery below the permitted level
                action[1] = -self._state[1]*C_tsM/100*(60/step_size)   
                self._state[1] = 0
            else:

                self._state[1] = turn_battery_to_int(self._state[1]+thermal_charge_percent)


            if thermal_charge_percent*(C_tsM/100) < -Therm_forecast[int(self._state[2])]*(60/step_size):     #If the agent wants to discharge more kWh than allowed
                action[1] = 1/eta_D_ts*(-Therm_forecast[int(self._state[2])]*(60/step_size))


                self._state[1] = original_thermal_soc_store

                self._state[1] = turn_battery_to_int(self._state[1]+100*((self.charge_discharge_thermal(action, previous_thermal))/C_tsM))     #TODO  Add battery into int

        else:
            if self._state[1]+thermal_charge_percent > 100:  # Checks if the action would charge the battery above 100%
                action[1] = (100 - self._state[1])*C_tsM/100*(60/step_size)        # Reduces such action to the highest possible to reach 100%
                self._state[1] = 100                

            else:
                self._state[1] = turn_battery_to_int(self._state[1]+thermal_charge_percent)
        


        if self._start_time == 1410:
            value_function_value = 0    #Initial agent has no one to look up to
        else:
            # value_func_path = os.getenv("value_function_path")

            input_value = self._start_time + 30
            value_function_value = get_value_function_value(self, self._environment_agent)
            #print(value_function_value)
            # value_func_path = f"{value_func_path}/{name}"
            # # Open the file in read mode
            # with open(value_func_path, "r") as file:
            #     # Read each line and strip newline characters
            #     value_function_list = [line.strip() for line in file]

            # value_function_value = float(value_function_list[int(self._state[0])-1])
 
        self._state[2] = self._state[2] + 1
        # if self._state[2] == int(end_time/step_size):  
        self._episode_ended = True
        

       
        
        # Terminal step value: should only be location dependent!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
        if self._episode_ended and self._state[2]==48: 
            return ts.termination(np.array([self._state], dtype=np.float32), 
            reward = self.running_cost(action, previous_electric, previous_thermal) +\
                ((self._state[0])*C_esM/100 + (self._state[1])*C_tsM/100)* Elec_Sell[int(self._state[2])] ) # Matching (20)
        elif self._episode_ended and self._state[2] < 48:
            return ts.termination(np.array([self._state], dtype=np.float32), reward = self.running_cost(action, previous_electric, previous_thermal) + value_function_value) 


environment = SolarEnvironment()        
# Discretizing action space
# Wrapping in a tenssorflow environment

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


#########################################################################################################################################################################################################
if __name__ =='__main__':
    environment = SolarEnvironment()
    
    print(utils.validate_py_environment(environment, episodes=5))

    load_dotenv("./envs_sac/simple_sac.env")


    time_step = environment.reset()
    print(time_step)

    cumulative_reward = time_step.reward

    for i in range(0, 48):
        charge = [0, 0]
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






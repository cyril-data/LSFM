import gym


import random
import numpy as np
import datetime as dt
import math

import pandas as pd
import seaborn as sns
sns.set()


import matplotlib.pyplot as plt
from .gridworld import SimpleGrid

from tensorflow import keras

class custom_env:

    def __init__(self, env_name, param={}):
        # Initialize atributes
        self.env_name = env_name    # name of environment
        self.param = param          # dictionarry of environment parameters

        if (self.env_name == "SimpleGrid"):
            self.env = SimpleGrid(self.param["grid_size"],
                                  block_pattern=self.param["pattern"],
                                  obs_mode=self.param["obs_mode"])

            self.state_space = gym.spaces.Discrete(4)
            self.env.reset(
                agent_pos=self.param["agent_pos"], 
                goal_pos=self.param["goal_pos"],
                reward_pos=self.param["reward_pos"],
                reward_minmax = self.param["reward_minmax"],
                reward_user_defined = self.param["reward_user_defined"]
            )

            self.state_type = "Discrete"
            self.observation_space = gym.spaces.Discrete(
                self.param["grid_size"]**2)
            
            self.action_space = gym.spaces.Discrete(4)
            
        if (self.env_name.split("_")[0] == "gym"):
            gym_name = "_".join(self.env_name.split("_")[1:])
            self.env = gym.make(gym_name)
            self.action_space = self.env.action_space
            if (type(self.env.observation_space) == gym.spaces.discrete.Discrete):
                self.observation_space = self.env.observation_space
                self.state_type = "Discrete"
                                
            else:
                self.state_type = "Continue"

        observation = self.reset()
        
        if self.state_type == "Discrete":
            self._state_dim = self.observation_space.n
        else:
            self._state_dim = len(observation)

    def reset(self):
        
        
        if self.state_type == "Discrete":
            if (self.env_name == "SimpleGrid"):
                self.env.reset(
                agent_pos=self.param["agent_pos"], 
                goal_pos=self.param["goal_pos"],
                reward_pos=self.param["reward_pos"] ,
                reward_minmax = self.param["reward_minmax"],
                reward_user_defined = self.param["reward_user_defined"]
                )
                                
                n = self.observation_space.n
                return_reset = np.array(keras.utils.to_categorical(
                    self.env.observation,   
                    num_classes=n)).reshape(n)
            else : 
                
                state_res = self.env.reset()
                
                n = self.observation_space.n
                return_reset = np.array(keras.utils.to_categorical(
                    state_res, 
                    num_classes=n)).reshape(n)
                
        else : 
            return_reset = self.env.reset()
        
        return return_reset

    def step(self, action):
        
                    
        if self.state_type == "Discrete":
            
            if (self.env_name == "SimpleGrid"):
                reward = self.env.step(action)
                done = self.env.done
                info = ""
                n = self.observation_space.n
                next_state = np.array(keras.utils.to_categorical(
                    self.env.observation, 
                    num_classes=n)).reshape(n)

                return_env = np.array(next_state), reward, done, info
            
            else :
            
                next_state, reward, done, info = self.env.step(action)

                n = self.observation_space.n
                next_state = np.array(keras.utils.to_categorical(
                    next_state, 
                    num_classes=n)).reshape(n)

                return_env = np.array(next_state), reward, done, info
            
        else:
            return_env = self.env.step(action)

        
        return return_env
        
        
    def render(self):
        if (self.env_name == "SimpleGrid"):
            
            fig, ax = plt.subplots()
            ax.imshow(self.env.grid)

            # Minor ticks
            ax.set_xticks(np.arange(-.5, (self._state_dim)**0.5 - 1, 1), minor=True)
            ax.set_yticks(np.arange(-.5, (self._state_dim)**0.5 - 1, 1), minor=True)

            # Gridlines based on minor ticks
            ax.grid(which='both', color='w', linestyle='-', linewidth=1.5)
        if (self.env_name.split("_")[0] == "gym"):
            self.env.render()

            
    def render_state(self, state) : 
        if (self.env_name == "SimpleGrid"):
            
            fig, ax = plt.subplots()
            ax.imshow(self.env.grid_state(state))

            # Minor ticks
            ax.set_xticks(np.arange(-.5, (self._state_dim)**0.5 - 1, 1), minor=True)
            ax.set_yticks(np.arange(-.5, (self._state_dim)**0.5 - 1, 1), minor=True)

            # Gridlines based on minor ticks
            ax.grid(which='both', color='w', linestyle='-', linewidth=1.5)
        
        
    def close(self):
        if (self.env_name.split("_")[0] == "gym"):
            self.env.close()

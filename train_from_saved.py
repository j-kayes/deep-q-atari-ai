import gym
import random
import numpy as np
import tensorflow as tf
import time
from agent import *

# Run this file using python to continue training a previously trained model:
if __name__ == "__main__":
    env = gym.make('Breakout-v0') 
    state_size = env.observation_space.shape[0]
    agent = Agent(env, state_size)
    
    agent.model = agent.load_model('latest_save.h5')
    agent.train_network()

    env.close()

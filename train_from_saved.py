import gym
import random
import numpy as np
import tensorflow as tf
import time
from agent import *

# Continues training a previously saved model:
if __name__ == "__main__":
    env = gym.make('Breakout-v0')
    state_size = env.observation_space.shape[0]
    agent = Agent(env, state_size)
    
    agent.model = agent.load_model('latest_model.h5')
    agent.train_network()

    env.close()

import gym
import random
import numpy as np
import tensorflow as tf
import time
from agent import *

if __name__ == "__main__":
    env = gym.make('Breakout-v0')
    state_size = env.observation_space.shape[0]
    agent = Agent(env, state_size)
    agent.train_network()

    env.close()

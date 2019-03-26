import gym
import random
import numpy as np
import tensorflow as tf
from agent import *

if __name__ == "__main__":
    env = gym.make('Pendulum-v0')
    state_size = env.observation_space.shape[0] 

    with Agent(env, state_size) as agent:
        agent.train_network()

    print('Done!')

import gym
import random
import numpy as np
import tensorflow as tf
import time
from agent import *

if __name__ == "__main__":
    env = gym.make('breakout-v1')
    state_size = env.observation_space.shape[0]
    agent = Agent(env, state_size)
    agent.train_network()

    env.close()

    '''
	for game in range(5):
		observation = env.reset()
		done = False
		total_reward = 0
		while(not done):
			env.render()
			action = env.action_space.sample()
			observation, reward, done, info = env.step(action)
			total_reward += reward
			time.sleep(1.0/60.0)
		print("Finished with total score {}".format(total_reward))
	'''

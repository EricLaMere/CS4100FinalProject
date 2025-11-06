import gymnasium as gym
import numpy as np
import sys


train_flag = 'train' in sys.argv
gui_flag = 'gui' in sys.argv


num_episodes = 1000
decay_rate = 0.99


# Define Q-Learning agent
def Q_learning(num_episodes=1000, decay_rate=0.999, gamma=0.9, epsilon=1): 

    Q_table = {}


# Train agent
if train_flag: 
    Q_table = None


# Evaluate agent
if not train_flag:
    Q_table = None
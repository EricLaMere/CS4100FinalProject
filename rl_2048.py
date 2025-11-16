import sys
import time
import pickle
import numpy as np
from tqdm import tqdm
from vis_2048 import *
import matplotlib
matplotlib.rcParams['agg.path.chunksize'] = 10000  # fix for large dataset plotting (1000000+)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

BOLD = '\033[1m'  # ANSI escape sequence for bold text
RESET = '\033[0m' # ANSI escape sequence to reset text formatting
train_flag = 'train' in sys.argv
gui_flag = 'gui' in sys.argv

setup(GUI=gui_flag)
env = game # Gym environment already initialized within vis_2048.py

# define hashing function for state representation
def hash_state(obs):
    # Convert to tuple and use Python's built-in hash
    return hash(tuple(obs.flatten()))

# define Q-Learning agent
def Q_learning(num_episodes=1000, decay_rate=0.999, gamma=0.9, epsilon=1): 

    # initialize Q-table
    Q_table = {}

    # count number of updates
    N_table = {}

    # intialize epsilon decay
    current_epsilon = epsilon
    
    # track training metrics
    episode_rewards = []
    episode_lengths = []
    max_tiles_achieved = []

    # training loop
    for episode in tqdm(range(num_episodes), desc="Training"):
        # reset environment
        obs = env.reset()
        total_reward = 0
        steps = 0
        
        while not env.game_over and not env.reached_2048 and steps < env.max_steps:
            # get current state
            state = hash_state(obs)
            
            # initalize state if not seen before
            if state not in Q_table:
                Q_table[state] = np.zeros(env.action_space.n)
            
            # epsilon-greedy action selection
            if np.random.random() < current_epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_table[state])
            
            # take action
            next_obs, reward, done, info = env.step(action)
            next_state = hash_state(next_obs)
            
            # calculate reward and update counters
            total_reward += reward
            steps += 1
            
            # initialize next state if not seen before
            if next_state not in Q_table:
                Q_table[next_state] = np.zeros(env.action_space.n)
            
            # Q-learning update
            if done:
                target = reward
            else:
                target = reward + gamma * np.max(Q_table[next_state])
            
            # update visit counter and learning rate
            state_action_key = (state, action)
            N_table[state_action_key] = N_table.get(state_action_key, 0) + 1
            eta = 1 / (1 + N_table[state_action_key])
            
            # update q-value
            Q_table[state][action] += eta * (target - Q_table[state][action])
            
            # move to next state
            obs = next_obs
            
            if done:
                break
        
        # record episode statistics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        max_tiles_achieved.append(env.get_max_tile())
        
        # decay epsilon
        current_epsilon *= decay_rate
        
    # final statistics
    print(f"\nTraining completed!")
    print(f"Total states discovered: {len(Q_table)}")
    print(f"Average final reward: {np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards):.2f}")
    print(f"Max tile achieved: {np.max(max_tiles_achieved)}")
    
    return Q_table


num_episodes = 1000
decay_rate = 0.999

# train agent
if train_flag: 
    Q_table = Q_learning(num_episodes=num_episodes, decay_rate=decay_rate, gamma=0.9, epsilon=1)

    # save Q-table
    with open('Q_table_'+str(num_episodes)+'_'+str(decay_rate)+'.pickle', 'wb') as handle:
        pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Evaluate agent
if not train_flag:
    Q_table = None
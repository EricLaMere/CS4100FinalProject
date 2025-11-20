''' same as rl_2048.py but the objective function is reaching the 2048 tile the fastest
    once 2048 tile is achieved the episode ends

    reward = 1000 - steps taken
    -100 reward if the agent loses the game before reaching 2048
    -1 reward for each step to encourage speed
'''

import sys
import time
import pickle
import numpy as np
from tqdm import tqdm
from vis_2048 import *
import matplotlib

matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

BOLD = '\033[1m'
RESET = '\033[0m'
train_flag = 'train' in sys.argv
gui_flag = 'gui' in sys.argv

setup(GUI=gui_flag)
env = game


def hash_state(obs):
    return hash(tuple(obs.flatten()))


def Q_learning(num_episodes=1000, decay_rate=0.999, gamma=0.9, epsilon=1):
    # initialize
    Q_table = {}
    N_table = {}
    current_epsilon = epsilon

    # track training metrics for speed objective
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    successful_episode_lengths = []

    for episode in tqdm(range(num_episodes), desc="Training"):
        obs = env.reset()
        steps = 0

        while not env.game_over and not env.reached_2048 and steps < env.max_steps:
            state = hash_state(obs)

            if state not in Q_table:
                Q_table[state] = np.zeros(env.action_space.n)

            # epsilon-greedy action selection
            if np.random.random() < current_epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_table[state])

            next_obs, _, done, info = env.step(action)
            next_state = hash_state(next_obs)
            steps += 1

            # penalize each step to encourage speed
            # give large reward for reaching 2048, penalty for each step taken
            if env.reached_2048:
                reward = 1000 - steps  # large reward minus steps taken
            elif env.game_over:
                reward = -100  # penalty for losing
            else:
                reward = -1  # small penalty for each step to encourage speed

            if next_state not in Q_table:
                Q_table[next_state] = np.zeros(env.action_space.n)

            # Q-learning update
            if done:
                target = reward
            else:
                target = reward + gamma * np.max(Q_table[next_state])

            state_action_key = (state, action)
            N_table[state_action_key] = N_table.get(state_action_key, 0) + 1
            eta = 1 / (1 + N_table[state_action_key])

            Q_table[state][action] += eta * (target - Q_table[state][action])

            obs = next_obs

            if done:
                break

        episode_lengths.append(steps)
        if env.reached_2048:
            success_count += 1
            successful_episode_lengths.append(steps)

        current_epsilon *= decay_rate

    # final statistics
    print(f"\nTraining completed!")
    print(f"Total states discovered: {len(Q_table)}")
    print(f"Success rate (reached 2048): {success_count}/{num_episodes} ({100 * success_count / num_episodes:.2f}%)")
    if successful_episode_lengths:
        print(f"Average steps to reach 2048 (successful episodes): {np.mean(successful_episode_lengths):.2f}")
        print(f"Best (fastest) run: {np.min(successful_episode_lengths)} steps")

    return Q_table


num_episodes = 1000
decay_rate = 0.999

if train_flag:
    Q_table = Q_learning(num_episodes=num_episodes, decay_rate=decay_rate, gamma=0.9, epsilon=1)

    with open('Q_table_' + str(num_episodes) + '_' + str(decay_rate) + '_speed_obj.pickle', 'wb') as handle:
        pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

if not train_flag:
    episode_lengths = []
    episode_times = []
    seen_states = set()
    total_actions = 0
    actions_using_Q = 0
    success_count = 0
    successful_episode_lengths = []

    filename = 'Q_table_' + str(num_episodes) + '_' + str(decay_rate) + '_speed_obj.pickle'
    input(
        f"\n{BOLD}Currently loading Q-table from " + filename + f"{RESET}.  \n\nPress Enter to confirm, or Ctrl+C to cancel and load a different Q-table file.\n(set num_episodes and decay_rate in Q_learning.py).")
    Q_table = np.load(filename, allow_pickle=True)

    for episode in tqdm(range(10000), desc="Evaluating"):
        obs = env.reset()
        steps = 0
        start_time = time.time()

        while not env.game_over and not env.reached_2048 and steps < env.max_steps:
            state = hash_state(obs)
            seen_states.add(state)

            if state in Q_table:
                action = np.argmax(Q_table[state])
                actions_using_Q += 1
            else:
                action = env.action_space.sample()

            total_actions += 1

            next_obs, reward, done, info = env.step(action)
            steps += 1
            obs = next_obs

            if done:
                break

        end_time = time.time()
        episode_lengths.append(steps)
        episode_times.append(end_time - start_time)

        if env.reached_2048:
            success_count += 1
            successful_episode_lengths.append(steps)

    # final statistics
    avg_length = np.mean(episode_lengths)
    total_time = np.sum(episode_times)

    print("Unique states in Q-table:", len(Q_table))
    print(f"Success rate (reached 2048): {success_count}/10000 ({100 * success_count / 10000:.2f}%)")
    if successful_episode_lengths:
        print(f"Average steps to reach 2048 (successful episodes): {np.mean(successful_episode_lengths):.2f}")
        print(f"Best (fastest) run: {np.min(successful_episode_lengths)} steps")
        print(f"Worst (slowest) run: {np.max(successful_episode_lengths)} steps")
    print(f"Average episode length (all episodes): {avg_length:.2f} steps")
    print(f"Total time to play 10,000 episodes: {total_time:.2f} seconds")

    unique_unseen_states = seen_states - set(Q_table.keys())
    print("Unique states unseen during training:", len(unique_unseen_states))

    percent_Q_usage = (actions_using_Q / total_actions) * 100
    print(f"Percentage of actions chosen using Q-table: {percent_Q_usage:.2f}%")
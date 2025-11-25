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
        reward = 0
        total_reward = 0

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

            total_reward += reward

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
        episode_rewards.append(total_reward)
        if env.reached_2048:
            success_count += 1
            successful_episode_lengths.append(steps)

        current_epsilon *= decay_rate

    # create high-resolution training rewards plot (used ai to help format the plot)
    plt.figure(figsize=(12, 8), dpi=300)
    episodes = range(1, len(episode_rewards) + 1)
    plt.plot(episodes, episode_rewards, alpha=0.3, color='lightblue', linewidth=0.5, label='Episode Rewards')
    # moving average
    window_size = max(1, num_episodes // 50)
    if len(episode_rewards) >= window_size:
        moving_avg = []
        for i in range(len(episode_rewards)):
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            moving_avg.append(np.mean(episode_rewards[start_idx:end_idx]))
        plt.plot(episodes, moving_avg, color='darkblue', linewidth=2,
                 label=f'Moving Average (window={window_size})')
    plt.title(
        f'Q-Learning Training Progress: Rewards per Episode\n(Episodes: {num_episodes}, ε decay: {decay_rate})',
        fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Episode', fontsize=14, fontweight='bold')
    plt.ylabel('Total Reward', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xlim(1, num_episodes)
    plt.tight_layout()
    plt.savefig(f'training_rewards_{num_episodes}_{decay_rate}_speed_obj.png', dpi=300, bbox_inches='tight',
                facecolor='white')


    # final statistics
    print(f"\nTraining completed!")
    print(f"Total states discovered: {len(Q_table)}")
    print(f"Success rate (reached 2048): {success_count}/{num_episodes} ({100 * success_count / num_episodes:.2f}%)")
    if successful_episode_lengths:
        print(f"Average steps to reach 2048 (successful episodes): {np.mean(successful_episode_lengths):.2f}")
        print(f"Best (fastest) run: {np.min(successful_episode_lengths)} steps")
    # Print average final reward and max tile achieved
    avg_final_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    print(f"Average final reward: {avg_final_reward:.2f}")
    # Track max tile achieved for all episodes
    all_max_tiles = [env.get_max_tile() for _ in range(num_episodes)] if not episode_rewards else [env.get_max_tile() for _ in range(len(episode_rewards))]
    print(f"Max tile achieved: {np.max(all_max_tiles)}")
    return Q_table, N_table


def prune_qtable(Q_table, N_table, min_visits=5):
    print(f"Pruning Q-table")
    print(f"Original states: {len(Q_table)}")

    visited_states = set()
    for (state, action), count in N_table.items():
        if count >= min_visits:
            visited_states.add(state)

    # only keep state-action pairs visited 5+ times
    pruned_Q = {state: actions for state, actions in Q_table.items() if state in visited_states}

    print(f"Pruned states: {len(pruned_Q)} ({len(pruned_Q) / len(Q_table) * 100:.1f}% kept)")

    return pruned_Q


num_episodes = 1000
decay_rate = 0.999

if train_flag:
    Q_table, N_table = Q_learning(num_episodes=num_episodes, decay_rate=decay_rate, gamma=0.9, epsilon=1)

    if num_episodes > 100000:
        # only look at states visited 5 times to save space in Q table
        Q_table = prune_qtable(Q_table, N_table, min_visits=5)

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
    rewards = []
    max_tiles_achieved = []

    filename = 'Q_table_' + str(num_episodes) + '_' + str(decay_rate) + '_speed_obj.pickle'
    input(
        f"\n{BOLD}Currently loading Q-table from " + filename + f"{RESET}.  \n\nPress Enter to confirm, or Ctrl+C to cancel and load a different Q-table file.\n(set num_episodes and decay_rate in Q_learning.py).")
    Q_table = np.load(filename, allow_pickle=True)

    for episode in tqdm(range(10000), desc="Evaluating"):
        obs = env.reset()
        steps = 0
        start_time = time.time()
        total_rewards_eval = 0

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
            total_rewards_eval += reward

            if done:
                break

        rewards.append(total_rewards_eval)
        max_tiles_achieved.append(env.get_max_tile())
        end_time = time.time()
        episode_lengths.append(steps)
        episode_times.append(end_time - start_time)

        if env.reached_2048:
            success_count += 1
            successful_episode_lengths.append(steps)
    avg_reward = np.mean(rewards)
    print(f"Average reward over 10,000 episodes: {avg_reward:.2f}")

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

    # Create evaluation visualizations
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    fig.suptitle(f'Evaluation Results: {num_episodes} Training Episodes, {decay_rate} Decay',
                 fontsize=16, fontweight='bold', y=1.02)

    # Subplot 1: rolling avg reward
    window = 100
    rolling_avg = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
    axes[0].plot(range(len(rewards)), rewards, alpha=0.2, color='lightblue',
                 linewidth=0.5, label='Episode Reward')
    axes[0].plot(range(len(rolling_avg)), rolling_avg, color='darkblue',
                 linewidth=2, label=f'{window}-Episode Moving Avg')
    axes[0].set_xlabel('Evaluation Episode', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Total Reward', fontsize=12, fontweight='bold')
    axes[0].set_title('Reward Progression During Evaluation', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # subplot 2: max tile distribution
    tile_counts = pd.Series(max_tiles_achieved).value_counts().sort_index()
    bars = axes[1].bar(range(len(tile_counts)), tile_counts.values,
                       color='coral', edgecolor='black', alpha=0.7)
    axes[1].set_xticks(range(len(tile_counts)))
    axes[1].set_xticklabels([int(tile) for tile in tile_counts.index], rotation=45)
    axes[1].set_xlabel('Max Tile Achieved', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title(f'Max Tile Distribution (Best: {np.max(max_tiles_achieved)})',
                      fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add percentage labels on bars
    for i, (bar, count) in enumerate(zip(bars, tile_counts.values)):
        height = bar.get_height()
        pct = (count / len(max_tiles_achieved)) * 100
        axes[1].text(bar.get_x() + bar.get_width() / 2., height,
                     f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    eval_plot_filename = f'evaluation_results_{num_episodes}_{decay_rate}_speed_obj.png'
    plt.savefig(eval_plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Evaluation plots saved to: {eval_plot_filename}")
    plt.close()

    


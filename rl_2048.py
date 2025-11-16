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
env = game # Gym environment already initialized within vis_gym.py


# Define Q-Learning agent
def Q_learning(num_episodes=1000, decay_rate=0.999, gamma=0.9, epsilon=1): 

    Q_table = {}


# Train agent
if train_flag: 
    Q_table = None


# Evaluate agent
if not train_flag:
    Q_table = None
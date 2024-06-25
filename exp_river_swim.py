
from utils import buildRiverSwim_patrol2, cumulative_rewards
from tqdm import tqdm
import time
import pdb
from learners.UCBVI_RM import UCBVI_RM
from learners.UCBVI_CP import UCBVI_CP
import matplotlib.pyplot as plt
import os
import csv

def run_exp(test_name = "river_swim_patrol"):
    epi_len = 15
    num_epi = 10000
    num_states = 5
    save_data = False
    env, n_states, n_actions = buildRiverSwim_patrol2(nbStates=num_states, rightProbaright=0.6, rightProbaLeft=0.05, rewardL=0.005,
                                               rewardR=1.)
    #learner = UCBVI_RM(n_states, n_actions, epi_len, delta = 0.05, K = num_epi, RM=env.rewardMachine)
    nQ = env.rewardMachine.n_states
    learner = UCBVI_CP(nQ, n_states, n_actions, epi_len, delta = 0.05, K = num_epi, rm_rewards=env.rewardMachine.rewards)
    cumulative_reward = cumulative_rewards(env, learner, len_horizon=epi_len)

    reward_per_episode = [total_reward[-1] for total_reward in cumulative_reward]
    chunk_size = 500
    total_reward_per_chunk = [sum(reward_per_episode[i: i + chunk_size]) for i in range(0, len(reward_per_episode),
                                                                                         chunk_size)]

    plt.plot(total_reward_per_chunk,  marker='o', linestyle='-', color='b')
    plt.title(learner.name())
    plt.show()

    # if data is requested to be saved:
    if save_data is True:
        data_name = f"data/river_swim_patrol_{num_states}states_{epi_len}h_{num_epi}K.csv"

        directory = os.getcwd()

        file_path = os.path.join(directory, data_name)

        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Rewards'])  # Write a header
            for reward in cumulative_reward:
                writer.writerow([reward])
    #print(len(cumulative_reward))
    #print(type(cumulative_reward))

    #pdb.set_trace()
if __name__ == "__main__":
    run_exp(test_name = "river_swim_patrol")

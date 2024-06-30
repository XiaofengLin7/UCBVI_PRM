from utils import buildRiverSwim_patrol2, buildFlower, cumulative_rewards_v1, cumulative_rewards_v2
from tqdm import tqdm
import numpy as np
import time
import pdb
from learners.UCBVI_RM import UCBVI_RM
from learners.UCBVI_CP import UCBVI_CP
from learners.UCRL_RM import UCRL2_RM
import matplotlib.pyplot as plt
import os
import csv
import multiprocessing as mp


def run_exp(test_name="river_swim_patrol"):
    save_data = True
    draw_data = False
    epi_len = 10
    num_epi = 5  # K
    num_states = 8  # can also be sizeB
    num_exp = 8
    if test_name == "river_swim_patrol":
        env, n_states, n_actions = buildRiverSwim_patrol2(nbStates=num_states, rightProbaright=0.6, rightProbaLeft=0.05,
                                                          rewardL=0.005,
                                                          rewardR=1.)
    elif test_name == "flower":
        env, n_states, n_actions = buildFlower(sizeB=num_states, delta=0.2)
    # nQ = env.rewardMachine.n_states
    # learner = UCBVI_CP(nQ, n_states, n_actions, epi_len, delta = 0.05, K = num_epi, rm_rewards=env.rewardMachine.rewards)
    # learner = UCRL2_RM(n_states, n_actions, epi_len, delta = 0.05, K = num_epi, RM=env.rewardMachine)
    result_1 = [[] for _ in range(num_exp)]
    result_2 = [[] for _ in range(num_exp)]
    learner_2 = UCRL2_RM(n_states, n_actions, epi_len, delta=0.05, K=num_epi, RM=env.rewardMachine)
    learner_1 = UCBVI_RM(n_states, n_actions, epi_len, delta=0.05, K=num_epi, RM=env.rewardMachine)
#    for i in range(num_exp):
#        print(f"\n Experiment {i + 1} begins.")
#        result_1[i] = cumulative_rewards_v1(env, learner_1, len_horizon=epi_len)
 #       result_2[i] = cumulative_rewards_v2(env, learner_2, len_horizon=epi_len)

    # Create a list of argument tuples for each experiment
    args = [(env, learner_1, learner_2, epi_len, i) for i in range(num_exp)]

    # Create a pool of worker processes
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Use starmap to apply data_collect to each tuple of arguments in parallel
        results = pool.starmap(data_collect, args)

    # Store the results in the appropriate arrays
    for i, res1, res2 in results:
        result_1[i] = res1
        result_2[i] = res2

    if draw_data:
        # TODO: get mean and std and draw.
        chunk_size = 500
        reward_per_episode_1 = [total_reward[-1] for total_reward in cumulative_reward_1]
        total_reward_per_chunk_1 = [sum(reward_per_episode_1[i: i + chunk_size]) for i in
                                    range(0, len(reward_per_episode_1), chunk_size)]

        reward_per_episode_2 = [total_reward[-1] for total_reward in cumulative_reward_2]
        total_reward_per_chunk_2 = [sum(reward_per_episode_2[i: i + chunk_size]) for i in
                                    range(0, len(reward_per_episode_2), chunk_size)]

        plt.plot(total_reward_per_chunk_1, marker='o', linestyle='-', color='b', label=learner_1.name())
        plt.plot(total_reward_per_chunk_2, marker='o', linestyle='-', color='r', label=learner_2.name())
        # plt.title(f"{num_epi} episodes {epi_len} horizon {num_states} states")
        plt.legend()
        plt.show()

    # if data is requested to be saved:
    if save_data is True:
        # data_name = "data/" + test_name + f"{num_states}states_{epi_len}h_{num_epi}K.csv"
        data_1 = np.array(result_1)
        data_2 = np.array(result_2)
        dir_name = "data/"
        data_name = test_name + f"_{num_states}states_{epi_len}h_{num_epi}K_{num_exp}runs.npz"
        data_path = os.path.join(dir_name, data_name)
        if os.path.exists(data_path):
            raise ValueError(f"A file with the name '{data_name}' already exists in the directory '{dir_name}'.")
        else:
            np.savez(dir_name + data_name, data_1=data_1, data_2=data_2)

        # directory = os.getcwd()

        # file_path = os.path.join(directory, data_name)

        # with open(file_path, 'w', newline='') as file:
        #    writer = csv.writer(file)
        #    writer.writerow(['Rewards'])  # Write a header
        #    for reward in cumulative_reward:
        #        writer.writerow([reward])
    # print(len(cumulative_reward))
    # print(type(cumulative_reward))


def data_collect(env, learner_1, learner_2, epi_len, i):
    print(f"Starting {i} run...\n")
    res1 = cumulative_rewards_v1(env, learner_1, len_horizon=epi_len)
    res2 = cumulative_rewards_v2(env, learner_2, len_horizon=epi_len)
    return i, res1, res2


if __name__ == "__main__":
    run_exp(test_name="flower")

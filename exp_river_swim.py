from utils import buildRiverSwim_patrol2, buildFlower, cumulative_rewards_v1, cumulative_rewards_v2
from tqdm import tqdm
import time
import pdb
from learners.UCBVI_RM import UCBVI_RM
from learners.UCBVI_CP import UCBVI_CP
from learners.UCRL_RM import UCRL2_RM
import matplotlib.pyplot as plt
import os
import csv


def run_exp(test_name="river_swim_patrol"):
    save_data = False

    epi_len = 10
    num_epi = 20000
    num_states = 10

    #env, n_states, n_actions = buildRiverSwim_patrol2(nbStates=num_states, rightProbaright=0.6, rightProbaLeft=0.05,
    #                                                  rewardL=0.005,
    #                                                  rewardR=1.)
    env, n_states, n_actions = buildFlower(sizeB = 5, delta = 0.2)
    learner_1 = UCBVI_RM(n_states, n_actions, epi_len, delta=0.05, K=num_epi, RM=env.rewardMachine)
    # nQ = env.rewardMachine.n_states
    # learner = UCBVI_CP(nQ, n_states, n_actions, epi_len, delta = 0.05, K = num_epi, rm_rewards=env.rewardMachine.rewards)
    # learner = UCRL2_RM(n_states, n_actions, epi_len, delta = 0.05, K = num_epi, RM=env.rewardMachine)
    chunk_size = 500

    cumulative_reward_1 = cumulative_rewards_v1(env, learner_1, len_horizon=epi_len)
    reward_per_episode_1 = [total_reward[-1] for total_reward in cumulative_reward_1]

    total_reward_per_chunk_1 = [sum(reward_per_episode_1[i: i + chunk_size]) for i in
                                range(0, len(reward_per_episode_1),
                                      chunk_size)]
    learner_2 = UCRL2_RM(n_states, n_actions, epi_len, delta=0.05, K=num_epi, RM=env.rewardMachine)
    cumulative_reward_2 = cumulative_rewards_v2(env, learner_2, len_horizon=epi_len)
    reward_per_episode_2 = [total_reward[-1] for total_reward in cumulative_reward_2]
    total_reward_per_chunk_2 = [sum(reward_per_episode_2[i: i + chunk_size]) for i in
                                range(0, len(reward_per_episode_2), chunk_size)]

    plt.plot(total_reward_per_chunk_1, marker='o', linestyle='-', color='b', label=learner_1.name())
    plt.plot(total_reward_per_chunk_2, marker='o', linestyle='-', color='r', label=learner_2.name())
    #plt.title(f"{num_epi} episodes {epi_len} horizon {num_states} states")
    plt.legend()
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
    # print(len(cumulative_reward))
    # print(type(cumulative_reward))

    # pdb.set_trace()


if __name__ == "__main__":
    run_exp(test_name="river_swim_patrol")

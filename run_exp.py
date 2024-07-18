from utils import *
from tqdm import tqdm
import numpy as np
import time
import pdb
from learners.UCBVI_RM import UCBVI_RM, UCBVI_PRM
from learners.UCBVI_CP import UCBVI_CP
from learners.Optimal_Player import Optimal_Player
from learners.UCRL_RM import UCRL2_RM
import matplotlib.pyplot as plt
import os
import csv
import multiprocessing as mp
import pickle


def run_exp(epi_len, num_epi, num_states, num_exp, test_name="river_swim_patrol"):
    save_data = True
    if test_name == "river_swim_patrol":
        env, n_states, n_actions = buildRiverSwim_patrol2(nbStates=num_states, rightProbaright=0.6, rightProbaLeft=0.05,
                                                          rewardL=0.005, rewardR=1., epi_len=epi_len)
        init_q, init_o = 0, 0
        print(f"Built River Swim MDP with {n_states} states and {epi_len} horizon...")
    elif test_name == "flower":
        env, n_states, n_actions = buildFlower(sizeB=num_states, delta=0.2, epi_len=epi_len)
        print(f"Built Multi-Task MDP with {n_states} states and {epi_len} horizon...")
        init_q, init_o = 0, 0
    elif test_name == "two_room_2corners":
        env, n_states, n_actions = buildGridworld_RM(num_states, num_states, epi_len, map_name="two_room_2corners")
        init_q = 0
        init_o = env.to_s([(int)(env.sizeX / 2), (int)(env.sizeY / 2)])
        print(f"Built two room MDP with {env.sizeX} x {env.sizeY} gridworld.")
    elif test_name == "warehouse":
        env, n_states, n_actions = buildWarehouse_PRM(num_states, num_states, epi_len, map_name="two_room")
        init_q = 0
        init_o = env.to_s([1, 1])
        print(f"Built warehouse with {env.sizeX} x {env.sizeY} gridworld.")
    elif test_name == "river_swim_patrol_prm":
        env, n_states, n_actions = buildRiverSwim_patrol2_PRM(nbStates=num_states, rightProbaright=0.6,
                                                        rightProbaLeft=0.05,rewardL=0.005, rewardR=1., epi_len=epi_len)
        init_q, init_o = 0, 0
        print(f"Built River Swim MDP with {n_states} states and {epi_len} horizon...")

    else:
        raise NameError("No such environment.")
    if save_data:
        print("Data will be saved after experiment completes.")
        with open(f'data/env/' + test_name + f'_{num_states}states_{epi_len}h.pkl', 'wb') as file:
            pickle.dump(env, file)
            print("Saving env object...")

    print(f"There will be {num_epi} episodes for each learner and {num_exp} times experiments will be run...")
    # nQ = env.rewardMachine.n_states
    # learner = UCBVI_CP(nQ, n_states, n_actions, epi_len, delta = 0.05, K = num_epi, rm_rewards=env.rewardMachine.rewards)
    # learner = UCRL2_RM(n_states, n_actions, epi_len, delta = 0.05, K = num_epi, RM=env.rewardMachine)
    result_1 = [[] for _ in range(num_exp)]
    result_2 = [[] for _ in range(num_exp)]
    optimal_res = [[] for _ in range(num_exp)]

    learner_1 = UCBVI_PRM(n_states, n_actions, epi_len, delta=0.05, K=num_epi, RM=env.rewardMachine)
    # learner_2 = UCRL2_RM(n_states, n_actions, epi_len, delta=0.05, K=num_epi, RM=env.rewardMachine)
    learner_2 = UCBVI_CP(env.rewardMachine.n_states, n_states, n_actions, epi_len, delta=0.05, K=num_epi, rm_rewards=env.rewardMachine.rewards)
    optimal_learner = Optimal_Player(env, K=num_epi)
    V_star = env.V_star[0, init_q, init_o]

    # Create a list of argument tuples for each experiment
    args = [(env, learner_1, learner_2, optimal_learner, epi_len, i) for i in range(num_exp)]

    # Create a pool of worker processes
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Use starmap to apply data_collect to each tuple of arguments in parallel
        results = pool.starmap(data_collect, args)

    # Store the results in the appropriate arrays
    for i, res1, res2, optimal_cumulative_reward in results:
        result_1[i] = res1
        result_2[i] = res2
        optimal_res[i] = optimal_cumulative_reward

    # if data is requested to be saved:
    if save_data is True:
        data_1 = np.array(result_1)
        data_2 = np.array(result_2)
        data_3 = np.array(optimal_res)
        P_star = env.P_star
        dir_name = "data/"
        data_name = test_name + f"_{num_states}states_{epi_len}h_{num_epi}K_{num_exp}runs.npz"
        data_path = os.path.join(dir_name, data_name)
        check_file = False
        if check_file and os.path.exists(data_path):
            raise ValueError(f"A file with the name '{data_name}' already exists in the directory '{dir_name}'.")
        else:
            np.savez(dir_name + data_name, data_1=data_1, data_2=data_2, data_3=data_3, V_star=V_star, P_star=P_star)


def data_collect(env, learner_1, learner_2, optimal_learner, epi_len, i):
    res1 = cumulative_rewards_v1(env, learner_1, len_horizon=epi_len)
    res2 = cumulative_rewards_v1(env, learner_2, len_horizon=epi_len)
    # res1 = 0
    # res2 = 0
    optimal = cumulative_rewards_v1(env, optimal_learner, len_horizon=epi_len)
    with open(f'data/learner/' + env.name() + f'_learner_1_{i}.pkl', 'wb') as file:
        pickle.dump(learner_1, file)
    with open(f'data/learner/' + env.name() + f'_learner_2_{i}.pkl', 'wb') as file:
        pickle.dump(learner_2, file)
    print(f"Finishing {i} run...\n")
    return i, res1, res2, optimal


if __name__ == "__main__":
    np.random.seed(None)
    epi_len = 5
    num_epi = 120000  # K
    num_states = 3
    num_exp = 8
    # test_name = "river_swim_patrol"
    # test_name = "flower"
    # test_name = "two_room_2corners"
    # test_name = "warehouse"
    test_name = "river_swim_patrol_prm"
    run_exp(epi_len, num_epi, num_states, num_exp, test_name=test_name)
    data_visualization = True
    visual_range = (0, num_epi)
    test_description = test_name+f'_{epi_len}_h_{num_states}_states'
    if data_visualization:
        data = np.load(f'data/' + test_name + f'_{num_states}states_{epi_len}h_{num_epi}K_{num_exp}runs.npz')
        result_1 = data['data_1']
        result_2 = data['data_2']
        result_star = data['data_3']
        V_star = data['V_star']
        # n_episodes = result_1.shape[1]
        with open(f'data/env/' + test_name + f'_{num_states}states_{epi_len}h.pkl', 'rb') as file:
            env = pickle.load(file)
        with open(f'data/learner/' + env.name() + f'_learner_1_{num_exp - 1}.pkl', 'rb') as file:
            learner_1 = pickle.load(file)
        with open(f'data/learner/' + env.name() + f'_learner_2_{num_exp - 1}.pkl', 'rb') as file:
            learner_2 = pickle.load(file)

        mean_cumu_reward_1, std_1 = calculate_cumu_reward_mean_std(result_1)
        mean_cumu_reward_2, std_2 = calculate_cumu_reward_mean_std(result_2)
        mean_cumu_reward_star, _ = calculate_cumu_reward_mean_std(result_star)

        k = np.arange(num_epi)

        regret_1 = -mean_cumu_reward_1 + (k + 1) * V_star  #
        regret_2 = -mean_cumu_reward_2 + (k + 1) * V_star  # mean_cumu_reward_star
        plt.figure(1)
        plt.plot(regret_1, marker='.', color='b', label=learner_1.name())
        plt.plot(regret_2, marker='.', color='r', label=learner_2.name())
        # plt.plot(k, (k+1)*V_star, marker='.', color='g', label='optimal')
        plt.fill_between(k, regret_1 - std_1, regret_1 + std_1, color='b', alpha=0.2)
        plt.fill_between(k, regret_2 - std_2, regret_2 + std_2, color='r', alpha=0.2)
        plt.ylabel('regret')
        plt.xlabel('episodes')
        plt.xlim(visual_range)
        plt.title(test_description)
        # plt.ylim(0, 0.2*n_episodes)
        plt.grid(True)
        plt.legend()
        plt.savefig('results/' + test_description + '_regret')

        plt.figure(2)
        plt.plot(k, mean_cumu_reward_1, marker='.', color='b', label=learner_1.name())
        plt.plot(k, mean_cumu_reward_2, marker='.', color='r', label=learner_2.name())
        plt.plot(k, (k + 1) * V_star, marker='.', color='g', label='optimal')
        plt.legend()
        plt.ylabel('cumulative reward')
        plt.xlabel('episodes')
        plt.xlim(visual_range)
        plt.title(test_description)
        plt.grid(True)
        plt.savefig('results/' + test_description + '_cumulative_reward')

        plt.figure(3)
        plt.plot(k, 1 / (k + 1) * mean_cumu_reward_1, marker='.', color='b', label=learner_1.name())
        plt.plot(k, 1 / (k + 1) * mean_cumu_reward_2, marker='.', color='r', label=learner_2.name())
        plt.plot(k, V_star * np.ones(num_epi), marker='.', color='g', label='optimal_1')
        em_cumu_reward_star = mean_cumu_reward_star
        em_cumu_reward_star[1:] = mean_cumu_reward_star[1:] - mean_cumu_reward_star[:-1]

        plt.plot(k, em_cumu_reward_star.mean() * np.ones(num_epi), marker='.', color='purple', label='optimal_2')
        plt.legend()
        plt.xlim(visual_range)
        plt.ylabel('averaged cumulative reward per episode')
        plt.xlabel('episodes')
        plt.title(test_description)
        plt.savefig('results/' + test_description + '_average_cumulative_reward')
        plt.show()

        print("plotting completes")

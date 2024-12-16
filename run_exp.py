from utils import *
from tqdm import tqdm
import numpy as np
import time
import pdb
from learners.UCBVI_RM import UCBVI_RM, UCBVI_PRM
from learners.UCBVI_CP import UCBVI_CP
from learners.Optimal_Player import Optimal_Player
from learners.UCRL_RM import UCRL2_RM
from learners.UCRL_B_RM import UCRL2_RM_Bernstein
import matplotlib.pyplot as plt
import os
import csv
import multiprocessing as mp
import pickle
from matplotlib.ticker import FuncFormatter


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
        env, n_states, n_actions = buildWarehouse_PRM(num_states, num_states, epi_len, map_name="warehouse")
        init_q = 0
        init_o = env.to_s([0, 0])
        print(f"Built warehouse with {env.sizeX} x {env.sizeY} gridworld.\n Slipperiness is {env.slippery}")
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
    # UCBVI_params = [1e-4,1e-3, 1e-2, 1e-1, 0.5, 1]
    # UCRL2_B_params = [ 1e-2, 0.1, 0.5, 0.75, 1]
    # UCRL2_L_params = [0.5, 0.75, 1]
    learners = []
    # for i in range(len(UCRL2_B_params)):
    #     learners.append(UCRL2_RM_Bernstein(n_states, n_actions, epi_len, delta=0.05, K=num_epi, RM=env.rewardMachine, distance_scale=UCRL2_B_params[i]))
    # for i in range(len(UCRL2_L_params)):
    #     learners.append(UCRL2_RM(n_states, n_actions, epi_len, delta=0.05, K=num_epi, RM=env.rewardMachine, distance_scale=UCRL2_L_params[i]))
    # for i in range(len(UCBVI_params)):
    #     learners.append(UCBVI_RM(n_states, n_actions, epi_len, delta=0.05, K=num_epi, RM=env.rewardMachine,
    #                                  bonus_scale=UCBVI_params[i]))

    ## RM
    learner_1 = UCBVI_RM(n_states, n_actions, epi_len, delta=0.05, K=num_epi, RM=env.rewardMachine,
                                     bonus_scale=0.001)
    learner_2 = UCRL2_RM(n_states, n_actions, epi_len, delta=0.05, K=num_epi, RM=env.rewardMachine, distance_scale=0.5)
    learner_3 = UCRL2_RM_Bernstein(n_states, n_actions, epi_len, delta=0.05, K=num_epi, RM=env.rewardMachine, distance_scale=0.1)
    learners.append(learner_1)
    learners.append(learner_2)
    learners.append(learner_3)

    ## PRM
    # learner_1 = UCBVI_PRM(n_states, n_actions, epi_len, delta=0.05, K=num_epi, RM=env.rewardMachine, bonus_scale=0.001)
    # learner_2 = UCBVI_CP(env.rewardMachine.n_states, n_states, n_actions, epi_len, delta=0.05, K=num_epi, rm_rewards=env.rewardMachine.rewards, bonus_scale=0.001)
    #     # learners.append(UCBVI_RM(n_states, n_actions, epi_len, delta=0.05, K=num_epi, RM=env.rewardMachine, bonus_scale=UCBVI_params[i]))
    # learners.append(learner_1)
    # learners.append(learner_2)

    num_learners = len(learners)
    result = [[[] for _ in range(num_exp)] for _ in range(num_learners)]
    V_star = env.V_star[0, init_q, init_o]

    # Create a list of argument tuples for each experiment
    args = [(env, learners, epi_len, i) for i in range(num_exp)]

    # Create a pool of worker processes
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Use starmap to apply data_collect to each tuple of arguments in parallel
        result_one_run = pool.starmap(data_collect, args)

    # Store the results in the appropriate arrays
    for i, cumu_rewards in result_one_run:
        for k in range(num_learners):
            result[k][i] = cumu_rewards[k]
        # result_1[i] = res1
        # result_2[i] = res2
        # optimal_res[i] = optimal_cumulative_reward

    # if data is requested to be saved:
    if save_data is True:
        # data_1 = np.array(result[0][:])
        # data_2 = np.array(result[1][:])
        all_cumu_rewards = np.array(result)
        # data_3 = np.array(optimal_res)
        P_star = env.P_star
        dir_name = "data/"
        data_name = test_name + f"_{num_states}states_{epi_len}h_{num_epi}K_{num_exp}runs.npz"
        data_path = os.path.join(dir_name, data_name)
        check_file = False
        if check_file and os.path.exists(data_path):
            raise ValueError(f"A file with the name '{data_name}' already exists in the directory '{dir_name}'.")
        else:
            np.savez(dir_name + data_name, all_cumu_rewards=all_cumu_rewards,
                     V_star=V_star, P_star=P_star)


def data_collect(env, learners, epi_len, i):
    num_learners = len(learners)
    cumu_rewards = [[] for _ in range(num_learners)]
    for k in range(num_learners):
        cumu_rewards[k] = cumulative_rewards_v1(env, learners[k], len_horizon=epi_len)
        with open(f'data/learner/' + env.name()+ f'_learner_{k+1}_{i}.pkl', 'wb') as file:
            pickle.dump(learners[k], file)
    print(f"Finishing {i} run...\n")
    return i, cumu_rewards

if __name__ == "__main__":
    np.random.seed(None)
    epi_len = 10
    num_epi = 10000  # K
    num_states = 5
    num_exp = 8
    test_name = "river_swim_patrol"
    # test_name = "flower"
    # test_name = "two_room_2corners"
    # test_name = "warehouse"
    # test_name = "river_swim_patrol_prm"
    run_exp(epi_len, num_epi, num_states, num_exp, test_name=test_name)
    data_visualization = True
    visual_range = (0, num_epi)
    test_description = test_name+f'_{epi_len}_h_{num_states}_states'
    if data_visualization:
        data = np.load(f'data/' + test_name + f'_{num_states}states_{epi_len}h_{num_epi}K_{num_exp}runs.npz')
        all_cumu_rewards = data['all_cumu_rewards']
        # result_star = data['data_3']
        num_learners = all_cumu_rewards.shape[0]
        V_star = data['V_star']
        # n_episodes = result_1.shape[1]
        with open(f'data/env/' + test_name + f'_{num_states}states_{epi_len}h.pkl', 'rb') as file:
            env = pickle.load(file)
        learners = []
        for i in range(num_learners):
            with open(f'data/learner/' + env.name() + f'_learner_{i+1}_{num_exp - 1}.pkl', 'rb') as file:
                learners.append(pickle.load(file))

        plot_results(all_cumu_rewards, learners, V_star, test_description)


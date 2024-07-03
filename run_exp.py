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
import pickle


def run_exp(test_name="river_swim_patrol"):
    save_data = True
    if save_data:
        print("Data will be saved after experiment completes.")
    epi_len = 6
    num_epi = 500000  # K
    num_states = 3  # can also be sizeB
    num_exp = 8
    if test_name == "river_swim_patrol":
        env, n_states, n_actions = buildRiverSwim_patrol2(nbStates=num_states, rightProbaright=0.6, rightProbaLeft=0.05,
                                                          rewardL=0.005, rewardR=1., epi_len=epi_len)
    elif test_name == "flower":
        env, n_states, n_actions = buildFlower(sizeB=num_states, delta=0.2, epi_len=epi_len)
    # nQ = env.rewardMachine.n_states
    # learner = UCBVI_CP(nQ, n_states, n_actions, epi_len, delta = 0.05, K = num_epi, rm_rewards=env.rewardMachine.rewards)
    # learner = UCRL2_RM(n_states, n_actions, epi_len, delta = 0.05, K = num_epi, RM=env.rewardMachine)
    result_1 = [[] for _ in range(num_exp)]
    result_2 = [[] for _ in range(num_exp)]
    transition_1 = [[] for _ in range(num_exp)]
    policy_set_1 = [[] for _ in range(num_exp)]
    policy_set_2 = [[] for _ in range(num_exp)]

    learner_1 = UCBVI_RM(n_states, n_actions, epi_len, delta=0.05, K=num_epi, RM=env.rewardMachine)
    learner_2 = UCRL2_RM(n_states, n_actions, epi_len, delta=0.05, K=num_epi, RM=env.rewardMachine)

    V_star = env.V_star[0, 0, 0]

    # Create a list of argument tuples for each experiment
    args = [(env, learner_1, learner_2, epi_len, i) for i in range(num_exp)]

    # Create a pool of worker processes
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Use starmap to apply data_collect to each tuple of arguments in parallel
        results = pool.starmap(data_collect, args)

    # Store the results in the appropriate arrays
    for i, res1, res2, learned_policy_1, learned_policy_2, p_1 in results:
        result_1[i] = res1
        result_2[i] = res2
        policy_set_1[i] = learned_policy_1
        policy_set_2[i] = learned_policy_2
        transition_1[i] = p_1

    # if data is requested to be saved:
    if save_data is True:
        # data_name = "data/" + test_name + f"{num_states}states_{epi_len}h_{num_epi}K.csv"
        data_1 = np.array(result_1)
        data_2 = np.array(result_2)
        policy_set_1_array = np.array(policy_set_1)
        policy_set_2_array = np.array(policy_set_2)
        transition_set_1 = np.array(transition_1)
        optimal_policy = env.optimal_policy
        P_star = env.P_star
        dir_name = "data/"
        data_name = test_name + f"_{num_states}states_{epi_len}h_{num_epi}K_{num_exp}runs.npz"
        data_path = os.path.join(dir_name, data_name)
        if os.path.exists(data_path):
            raise ValueError(f"A file with the name '{data_name}' already exists in the directory '{dir_name}'.")
        else:
            np.savez(dir_name + data_name, data_1=data_1, data_2=data_2, policy_1=policy_set_1_array,
                     policy_2=policy_set_2_array, V_star=V_star, optimal_policy=optimal_policy, transition_set_1=
                     transition_set_1, P_star=P_star)

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

    res2 = cumulative_rewards_v2(env, learner_2, len_horizon=epi_len)
    res1 = cumulative_rewards_v1(env, learner_1, len_horizon=epi_len)
    #res1 = 0
    with open(f'data/learner/learner_1_{i}.pkl', 'wb') as file:
        pickle.dump(learner_1, file)
    with open(f'data/learner/learner_2_{i}.pkl', 'wb') as file:
        pickle.dump(learner_2, file)
    print(f"Finishing {i} run...\n")
    return i, res1, res2, learner_1.get_policy(), learner_2.get_policy(), learner_1.p


if __name__ == "__main__":
    #run_exp(test_name="river_swim_patrol")
    run_exp(test_name="flower")
    data_visualization = False
    if data_visualization:
        data = np.load('data/flower_3states_6h_10000K_8runs.npz')
        result_1 = data['data_1']
        result_2 = data['data_2']
        V_star = data['V_star']
        p_1 = data['transition_set_1']
        reward_per_episode_1 = result_1[:, :, -1]
        cumu_rewards_1 = np.cumsum(reward_per_episode_1, axis=1)
        std_1 = np.std(cumu_rewards_1, axis=0)
        mean_1 = np.mean(result_1, axis=0)
        mean_reward_per_episode_1 = mean_1[:, -1]
        mean_cumu_reward_1 = np.cumsum(mean_reward_per_episode_1)

        reward_per_episode_2 = result_2[:, :, -1]
        cumu_rewards_2 = np.cumsum(reward_per_episode_2, axis=1)
        std_2 = np.std(cumu_rewards_2, axis=0)
        mean_2 = np.mean(result_2, axis=0)
        mean_reward_per_episode_2 = mean_2[:, -1]
        mean_cumu_reward_2 = np.cumsum(mean_reward_per_episode_2)

        k = np.arange(len(mean_reward_per_episode_2))

        regret_1 = -mean_cumu_reward_1 + (k+1)*(V_star) #0.85707315
        regret_2 = -mean_cumu_reward_2 + (k+1)*(V_star)
        plt.figure(1)
        plt.plot(regret_1, marker='.', color='b', label='UCBVI_RM')
        plt.plot(regret_2, marker='.', color='r', label='UCRL2_RM')
        #plt.plot(k, k*V_star, marker='.', color='g', label='optimal')
        plt.fill_between(k, regret_1 - std_1, regret_1 + std_1, color='b', alpha=0.2)
        plt.fill_between(k, regret_2 - std_2, regret_2 + std_2, color='r', alpha=0.2)
        plt.ylabel('regret')
        plt.legend()

        plt.figure(2)
        plt.plot(k, mean_cumu_reward_1, marker='.', color='b', label='UCBVI_RM')
        plt.plot(k, mean_cumu_reward_2, marker='.', color='r', label='UCRL2_RM')
        plt.plot(k, (k+1)*V_star, marker='.', color='g', label='optimal')
        plt.legend()
        plt.show()

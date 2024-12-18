import numpy as np
from gym.utils import seeding
import gym
from gym.envs.registration import register
from tqdm import tqdm
import pdb
import time
from matplotlib import pyplot as plt

def categorical_sample(prob_n, np_random):
    """
    Params: prob_n: probability for each element
    np_random: random number generator

    Return: sampled index
    """

    # prob_n = np.asarray(prob_n)
    # csprob_n = np.cumsum(prob_n)
    # return (csprob_n > np_random.rand()).argmax()
    prob_n = np.asarray(prob_n)
    k = np.arange(len(prob_n))
    return np.random.choice(k, p=prob_n)


def calculate_variance(prob_n: np.array, x: np.array) -> float:
    """
    :param prob_n:
    :param x:
    :return: variance
    """
    if abs(np.sum(prob_n) - 1) > 1e-5:
        pdb.set_trace()
        raise ValueError("Sum of probabilities is not 1")
    e_x = np.sum(np.multiply(prob_n, x))
    e_xx = np.sum(np.multiply(np.multiply(prob_n, x), x))
    var = e_xx - e_x ** 2
    if var < -1e-5:
        raise ValueError("The variance can't be negative.")
    elif -1e-5 <= var <= 0:
        return 0
    return var


def clip(x, range):
    """
    Params: x: a real number
    range: range to clip
    """
    return max(min(x, range[1]), range[0])


def calculate_cumu_reward_mean_std(all_reward_episodes_runs):
    """
    input: all_reward_episodes_runs: n_runs x n_episodes x epi_len
    output: mean_cumu_reward: averaged cumulative reward across runs
            std: corresponding standard deviation
    """
    reward_per_episode = all_reward_episodes_runs[:, :, -1]
    cumu_rewards = np.cumsum(reward_per_episode, axis=1)
    std = np.std(cumu_rewards, axis=0)
    mean_cumu_reward = np.mean(cumu_rewards, axis=0)

    return mean_cumu_reward, std


def value_iteration(P, R, epi_len):
    """
    :param P: nQ x nO x nA x nQ x nO
    :param R: nQ x nO x nA
    :param epi_len: H
    :return: V, Q
    """
    nQ = R.shape[0]
    nO = R.shape[1]
    nA = R.shape[2]
    V = np.zeros((epi_len + 1, nQ, nO), dtype=np.float64)
    Q = np.zeros((epi_len, nQ, nO, nA), dtype=np.float64)
    policy = np.zeros((epi_len, nQ, nO), dtype=int)
    for h in range(epi_len - 1, -1, -1):
        for q in range(nQ):
            for o in range(nO):
                for a in range(nA):
                    PV = np.sum(P[q, o, a, :, :] * V[h + 1, :, :])
                    Q[h, q, o, a] = PV + R[q, o, a]
                V[h, q, o] = np.max(Q[h, q, o, :])
                action_value = Q[h, q, o, :]
                action = np.random.choice(np.where(action_value == action_value.max())[0])
                # policy[h, q, o] = np.argmax(Q[h, q, o, :])
                policy[h, q, o] = action

    return Q, V, policy


def buildRiverSwim_patrol2(nbStates=5, max_steps=np.infty, reward_threshold=np.infty, rightProbaright=0.6,
                           rightProbaLeft=0.05, rewardL=0.1, rewardR=1., epi_len=10):
    register(
        id='RiverSwim_patrol2-v0',
        entry_point='environments.MDPRM_library:RiverSwim_patrol2',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'nbStates': nbStates, 'rightProbaright': rightProbaright, 'rightProbaLeft': rightProbaLeft,
                'rewardL': rewardL, 'rewardR': rewardR, 'epi_len': epi_len}
    )

    return gym.make('RiverSwim_patrol2-v0'), nbStates, 2


def buildFlower(sizeB, delta, epi_len, max_steps=np.infty, reward_threshold=np.infty):
    register(
        id='Flower_' + str(sizeB) + '-v0',
        entry_point='environments.MDPRM_library:Flower',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'sizeB': sizeB, 'delta': delta, 'epi_len': epi_len}
    )
    name = 'Flower_' + str(sizeB) + '-v0'
    return gym.make(name), 6, 2


def buildGridworld_RM(sizeX, sizeY, epi_len, map_name="2-room_1corner",
                      max_steps=np.infty, reward_threshold=np.infty):
    register(
        id='Gridworld-RM' + map_name + '-v0',
        entry_point='environments.MDPRM_library:RM_GridWorld',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'sizeX': sizeX, 'sizeY': sizeY, 'epi_len': epi_len, 'map_name': map_name}
    )
    g = gym.make('Gridworld-RM' + map_name + '-v0')
    return g, g.env.nS, 4

def buildWarehouse_PRM(sizeX, sizeY, epi_len, map_name="two_room",
                      max_steps=np.infty, reward_threshold=np.infty):
    register(
        id='Warehouse-PRM' + map_name + '-v0',
        entry_point='environments.MDPRM_library:Warehouse_PRM',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'sizeX': sizeX, 'sizeY': sizeY, 'epi_len': epi_len, 'map_name': map_name}
    )
    g = gym.make('Warehouse-PRM' + map_name + '-v0')
    return g, g.env.nS, 5


def buildRiverSwim_patrol2_PRM(nbStates=5, max_steps=np.infty, reward_threshold=np.infty, rightProbaright=0.6,
                           rightProbaLeft=0.05, rewardL=0.1, rewardR=1., epi_len=10):
    register(
        id='RiverSwim_patrol2_PRM-v0',
        entry_point='environments.MDPRM_library:RiverSwim_patrol2_PRM',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'nbStates': nbStates, 'rightProbaright': rightProbaright, 'rightProbaLeft': rightProbaLeft,
                'rewardL': rewardL, 'rewardR': rewardR, 'epi_len': epi_len}
    )

    return gym.make('RiverSwim_patrol2_PRM-v0'), nbStates, 2

def cumulative_rewards_v1(env, learner, len_horizon):
    cumulative_rewards = []

    for k in tqdm(range(learner.K), desc=learner.name()):
        # np.random.seed(42)
        learner.learn()
        observation = env.reset()
        learner.reset(observation)
        cur_epi_rewards = []
        cur_epi_cum_rewards = 0.0

        for t in range(len_horizon):
            cur_obs = observation
            cur_Q = env.rewardMachine.current_state
            action = learner.play(t, cur_Q, cur_obs)
            observation, reward, _, _ = env.step(action)
            cur_Q = env.rewardMachine.current_state
            learner.update(cur_Q, action, reward, observation, t)
            cur_epi_cum_rewards += reward
            cur_epi_rewards.append(cur_epi_cum_rewards)


        # print("Episode {}: cumulative reward is {}".format(k+1, cur_epi_cum_rewards))
        cumulative_rewards.append(cur_epi_rewards)
        # learner.learn()
    return cumulative_rewards
def plot_results(all_cumu_rewards, learners, V_star, test_description):
    color_set = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    # color_set = ['r','m', 'y','g', 'b', 'c',  'k']
    num_learners = len(learners)
    num_epi = all_cumu_rewards.shape[2]
    visual_range = (0, num_epi)
    mean_cumu_rewards = []
    std = []
    for i in range(num_learners):
        mean_cumu_reward, std_ = calculate_cumu_reward_mean_std(all_cumu_rewards[i])
        mean_cumu_rewards.append(mean_cumu_reward)
        std.append(std_)
    # plot the results
    k = np.arange(num_epi)
    regrets = []
    plt.rcParams.update({'font.size': 20})
    for i in range(num_learners):
        regret = -mean_cumu_rewards[i] + (k + 1) * V_star
        regrets.append(regret)
    plt.figure(1, figsize=(6.4, 4.8))
    for i in range(num_learners):
        plt.plot(k, regrets[i], marker='.', color=color_set[i], label=learners[i].name())
        plt.fill_between(k, regrets[i] - std[i], regrets[i] + std[i], color=color_set[i], alpha=0.2)
    plt.ylabel('Regret')
    plt.xlabel('Episodes')
    #plt.ylim((-50,1700))
    #plt.yticks(np.arange(0, 1501, 250))
    plt.xlim(visual_range)
    # plt.title(test_description)
    # plt.ylim(0, 150)
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.savefig('results/' + test_description + '_regret.pdf',  bbox_inches='tight')

    interval = 200
    mean_reward_per_episode = []

    for i in range(num_learners):
        mean_reward_per_episode_i = mean_cumu_rewards[i]
        mean_reward_per_episode_i[1:] = mean_cumu_rewards[i][1:] - mean_cumu_rewards[i][:-1]
        mean_reward_per_episode_i = mean_of_every_interval(mean_reward_per_episode_i, interval)
        mean_reward_per_episode.append(mean_reward_per_episode_i)

    plt.figure(2)
    k = np.arange(num_epi / interval)
    for i in range(num_learners):
        plt.plot(k * interval, mean_reward_per_episode[i], marker='.', color=color_set[i], label=learners[i].name())
    plt.plot(k * interval, V_star * np.ones((int)(num_epi / interval)), marker='.', color=color_set[-1], label='V*')
    plt.legend(loc='best')
    plt.xlim(visual_range)
    plt.ylabel('Empirical return per episode')
    plt.xlabel('Episodes')
    # plt.title(test_description)
    plt.savefig('results/' + test_description + '_average_cumulative_reward.pdf')
    plt.show()
    print("plotting completes")

def mean_of_every_interval(array, interval):
    # Split the array into chunks of 10
    chunks = [array[i:i + interval] for i in range(0, len(array), interval)]

    # Calculate the mean of each chunk
    means = [np.mean(chunk) for chunk in chunks]

    return np.array(means)

if __name__ == '__main__':
    generator, seed = seeding.np_random(42)
    print(type(generator))
    print(seed)

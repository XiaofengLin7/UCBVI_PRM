import numpy as np
from gym.utils import seeding
import gym
from gym.envs.registration import register
from tqdm import  tqdm
import pdb
def categorical_sample(prob_n, np_random):
    """
    Params: prob_n: probability for each element
    np_random: random number generator

    Return: sampled index
    """

    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()

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
    var = e_xx - e_x**2
    if var < -1e-5:
        raise ValueError("The variance can't be negative.")
    elif -1e-5 <= var <= 0:
        return 0
    return var

def clip(x,range):
    """
    Params: x: a real number
    range: range to clip
    """
    return max(min(x,range[1]),range[0])

def buildRiverSwim_patrol2(nbStates=5, max_steps=np.infty,reward_threshold=np.infty,rightProbaright=0.6, rightProbaLeft=0.05, rewardL=0.1, rewardR=1.):
    register(
        id='RiverSwim_patrol2-v0',
        entry_point='environments.MDPRM_library:RiverSwim_patrol2',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'nbStates': nbStates, 'rightProbaright': rightProbaright, 'rightProbaLeft': rightProbaLeft,
                'rewardL': rewardL, 'rewardR':rewardR, }
    )

    return gym.make('RiverSwim_patrol2-v0'), nbStates, 2

def cumulative_rewards_v1(env, learner, len_horizon):
    cumulative_rewards = []
    learner.update_Q()
    for k in tqdm(range(learner.K)):
        observation = env.reset()
        learner.reset(observation)
        cur_epi_rewards = []
        cur_epi_cum_rewards = 0.0
        for t in range(len_horizon):
            cur_obs = observation
            cur_Q = env.rewardMachine.current_state
            state = (cur_Q, cur_obs)
            action = learner.play(t, cur_Q, cur_obs)

            observation, reward, done, info = env.step(action)
            learner.update(cur_Q, action, reward, observation, t)
            cur_epi_cum_rewards += reward
            cur_epi_rewards.append(cur_epi_cum_rewards)
            #if env.rewardMachine.current_state == 1:
                #pdb.set_trace()
            #    pass

        #print("Episode {}: cumulative reward is {}".format(k+1, cur_epi_cum_rewards))
        cumulative_rewards.append(cur_epi_rewards)
        learner.learn()
    #pdb.set_trace()
    return cumulative_rewards



def cumulative_rewards_v2(env, learner, len_horizon):
    """
    util functions specifically for UCRL2 type of learning algorithms
    """
    cumulative_rewards = []
    for k in tqdm(range(learner.K)):
        observation = env.reset()
        learner.reset(observation)
        cur_epi_rewards = []
        cur_epi_cum_rewards = 0.0
        for t in range(len_horizon):
            cur_obs = observation
            cur_Q = env.rewardMachine.current_state
            state = (cur_Q, cur_obs)
            action = learner.play(t, cur_Q, cur_obs)

            observation, reward, done, info = env.step(action)
            learner.update(cur_Q, action, reward, observation, t)
            cur_epi_cum_rewards += reward
            cur_epi_rewards.append(cur_epi_cum_rewards)

        cumulative_rewards.append(cur_epi_rewards)

    #pdb.set_trace()
    return cumulative_rewards

if __name__ == '__main__':

    generator,seed = seeding.np_random(42)
    print(type(generator))
    print(seed)

import numpy as np

from utils import *
from environments.discreteMDP import DiscreteMDP
from environments.rewardMachine import RewardMachine

import scipy.stats as stat
import pdb
'''
Customized MDP environment and Reward Machine functions.
'''


def RM_riverSwim_patrol2(S):
    """
	Construct reward machine for river swim MDP, where reward will be only collected when visit two states of MDP.
	:param S: number of states
	:return:  events: O x A x O
			transitions:  Q x events
			rewards: Q x Q
	"""
    events = np.array([[None, None] for _ in range(S)])
    events[0, 0] = 0
    events[0, 1] = 0
    events[S - 1, 0] = 1
    events[S - 1, 1] = 1
    transitions = np.array([[0, 1], [0, 1]])
    rewards = np.array([[0, 1], [0, 0]])

    return events, transitions, rewards


class RiverSwim_patrol2(DiscreteMDP):
    def __init__(self, nbStates, rightProbaright=0.6, rightProbaLeft=0.05, rewardL=0.1,
                 rewardR=1.):  # , ergodic=False):
        self.nS = nbStates
        self.nA = 2
        self.states = range(0, nbStates)
        self.actions = range(0, self.nA)
        self.nameActions = ["R", "L"]

        self.startdistribution = np.zeros((self.nS))
        self.startdistribution[0] = 1.
        self.rewards = {}
        self.P = {}
        self.transitions = {}
        # Initialize a randomly generated MDP
        for s in self.states:
            self.P[s] = {}
            self.transitions[s] = {}
            # GOING RIGHT
            self.transitions[s][0] = {}
            self.P[s][0] = []  # 0=right", 1=left
            li = self.P[s][0]
            prr = 0.
            if (s < self.nS - 1) and (s > 0):
                li.append((rightProbaright, s + 1, False))
                self.transitions[s][0][s + 1] = rightProbaright
                prr = rightProbaright
            elif (s == 0):  # To have 0.6 on the leftmost state
                li.append((0.6, s + 1, False))
                self.transitions[s][0][s + 1] = 0.6
                prr = 0.6
            prl = 0.
            if (s > 0) and (s < self.nS - 1):  # MODIFY HERE FOR THE RIGTHMOST 0.95 and leftmost 0.35
                li.append((rightProbaLeft, s - 1, False))
                self.transitions[s][0][s - 1] = rightProbaLeft
                prl = rightProbaLeft
            elif s == self.nS - 1:  # To have 0.6 and 0.4 on rightmost state
                li.append((0.4, s - 1, False))
                self.transitions[s][0][s - 1] = 0.4
                prl = 0.4
            li.append((1. - prr - prl, s, False))
            self.transitions[s][0][s] = 1. - prr - prl

            # GOING LEFT
            # if ergodic:
            #	pll = 0.95
            # else:
            #	pll = 1
            self.P[s][1] = []  # 0=right", 1=left
            self.transitions[s][1] = {}
            li = self.P[s][1]
            if (s > 0):
                li.append((1., s - 1, False))
                self.transitions[s][1][s - 1] = 1.
            else:
                li.append((1., s, False))
                self.transitions[s][1][s] = 1.

            self.rewards[s] = {}
            if (s == self.nS - 1):
                self.rewards[s][0] = stat.norm(loc=rewardR, scale=0.)
            else:
                self.rewards[s][0] = stat.norm(loc=0., scale=0.)
            if (s == 0):
                self.rewards[s][1] = stat.norm(loc=rewardL, scale=0.)
            else:
                self.rewards[s][1] = stat.norm(loc=0., scale=0.)

        # print("Rewards : ", self.rewards, "\nTransitions : ", self.transitions)

        e, t, r = RM_riverSwim_patrol2(nbStates)
        self.rewardMachine = RewardMachine(e, t, r)
        # pdb.set_trace()
        super(RiverSwim_patrol2, self).__init__(self.nS, self.nA, self.P, self.rewards, self.startdistribution)

    def reset(self):
        self.rewardMachine.reset()
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        return self.s

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, d = transitions[i]
        event = self.rewardMachine.events[s, a]
        r = self.rewardMachine.next_step(event)
        self.s = s
        self.lastaction = a
        return (s, r, d, "")


if __name__ == "__main__":
    Test = True

    if Test:

        env, nbS, nbA = buildRiverSwim_patrol2(nbStates=20, rightProbaright=0.4, rightProbaLeft=0.05, rewardL=0.005,
                                               rewardR=1.)




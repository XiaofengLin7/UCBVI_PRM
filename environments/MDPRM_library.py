import numpy as np

from utils import *
from environments.discreteMDP import DiscreteMDP
from environments.rewardMachine import RewardMachine, RewardMachine2

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
    def __init__(self, nbStates, epi_len, rightProbaright=0.6, rightProbaLeft=0.05, rewardL=0.1,
                 rewardR=1.):  # , ergodic=False):
        self.nO = nbStates
        self.nA = 2
        self.states = range(0, nbStates)
        self.actions = range(0, self.nA)
        self.nameActions = ["R", "L"]

        self.startdistribution = np.zeros((self.nO))
        self.startdistribution[0] = 1.
        self.rewards = {}
        self.P = {}
        self.transitions = {}
        self.epi_len = epi_len
        # Initialize a randomly generated MDP
        for s in self.states:
            self.P[s] = {}
            self.transitions[s] = {}
            # GOING RIGHT
            self.transitions[s][0] = {}
            self.P[s][0] = []  # 0=right", 1=left
            li = self.P[s][0]
            prr = 0.
            if (s < self.nO - 1) and (s > 0):
                li.append((rightProbaright, s + 1, False))
                self.transitions[s][0][s + 1] = rightProbaright
                prr = rightProbaright
            elif (s == 0):  # To have 0.6 on the leftmost state
                li.append((0.6, s + 1, False))
                self.transitions[s][0][s + 1] = 0.6
                prr = 0.6
            prl = 0.
            if (s > 0) and (s < self.nO - 1):  # MODIFY HERE FOR THE RIGTHMOST 0.95 and leftmost 0.35
                li.append((rightProbaLeft, s - 1, False))
                self.transitions[s][0][s - 1] = rightProbaLeft
                prl = rightProbaLeft
            elif s == self.nO - 1:  # To have 0.6 and 0.4 on rightmost state
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
            if (s == self.nO - 1):
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
        self.Q_star = np.zeros((self.epi_len, self.rewardMachine.n_states, self.nO, self.nA), dtype=np.float64)
        self.V_star = np.zeros((self.epi_len+1, self.rewardMachine.n_states, self.nO), dtype=np.float64)
        self.optimal_policy = np.zeros((self.epi_len, self.rewardMachine.n_states, self.nO), dtype=int)
        self.optimal_player()
        # pdb.set_trace()
        super(RiverSwim_patrol2, self).__init__(self.nO, self.nA, self.P, self.rewards, self.startdistribution)

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
    def optimal_player(self):
        # get P and R, using dp to solve the optimal value function
        nQ = self.rewardMachine.n_states
        P = np.zeros((nQ, self.nO, self.nA, nQ, self.nO))
        R = np.zeros((nQ, self.nO, self.nA))
        # construct real P and R
        for q in range(nQ):
            for o in range(self.nO):
                for a in range(self.nA):
                    # event should be o x a x o
                    event = self.rewardMachine.events[o, a]
                    if event is not None:
                        next_q = self.rewardMachine.transitions[q, event]
                        R[q, o, a] = self.rewardMachine.rewards[q, event]
                        for z in self.transitions[o][a].keys():
                            #pdb.set_trace()
                            P[q, o, a, next_q, z] = self.transitions[o][a][z]
                    else:
                        next_q = q
                        for z in self.transitions[o][a].keys():
                            P[q, o, a, next_q, z] = self.transitions[o][a][z]

        # Value Iteration for given P and R
        for h in range(self.epi_len-1, -1, -1):
            for q in range(nQ):
                for o in range(self.nO):
                    for a in range(self.nA):
                        PV = np.sum(P[q, o, a, :, :]*self.V_star[h+1, :, :])
                        self.Q_star[h, q, o, a] = PV + R[q, o, a]

                    self.optimal_policy[h, q, o] = np.argmax(self.Q_star[h, q, o, :])
                    self.V_star[h, q, o] = np.max(self.Q_star[h, q, o, :])
        print("value iteration finishes.")


def RM_Flower(sizeB):
    liste_1_step = [1, 2, 4, 5, 7, 8, 10, 11]
    liste_back_center = [3, 6, 9, 12]
    events = np.array([[None, None] for _ in range(6)])
    events[1, 0] = 0  # small 1
    events[2, 0] = 1  # small 2
    events[3, 0] = 2  # small 3
    events[4, 0] = 3  # small 4
    events[5, 0] = 4  # Big loop
    transitions = []
    nQ = 1 + 4 * 3 + sizeB  # central state, 4 small loop, one big loop
    rewards = []
    for q in range(nQ):
        reward = [0 for _ in range(5)]
        if q == 0:
            transition = [0 for _ in range(5)]
            transition[0] = 1  # start small loop 1
            transition[1] = 4  # start small loop 2
            transition[2] = 7  # start small loop 3
            transition[3] = 10  # start small loop 4
            transition[4] = 13  # start big loop
        # small loop 1
        elif q in [1, 2]:
            transition = [q for _ in range(5)]
            transition[0] = q + 1
        elif q == 3:
            transition = [q for _ in range(5)]
            transition[0] = 0
            reward[0] = 1
        # small loop 2
        elif q in [4, 5]:
            transition = [q for _ in range(5)]
            transition[1] = q + 1
        elif q == 6:
            transition = [q for _ in range(5)]
            transition[1] = 0
            reward[1] = 1
        # small loop 3
        elif q in [7, 8]:
            transition = [q for _ in range(5)]
            transition[2] = q + 1
        elif q == 9:
            transition = [q for _ in range(5)]
            transition[2] = 0
            reward[2] = 1
        # small loop 4
        elif q in [10, 11]:
            transition = [q for _ in range(5)]
            transition[3] = q + 1
        elif q == 12:
            transition = [q for _ in range(5)]
            transition[3] = 0
            reward[3] = 1
        # BIG loop
        elif q < nQ - 1:
            transition = [q for _ in range(5)]
            transition[4] = q + 1
        else:  # nQ - 1
            transition = [q for _ in range(5)]
            transition[4] = 0
            reward[4] = 1
        transitions.append(transition)
        rewards.append(reward)
    transitions = np.array(transitions)
    rewards = np.array(rewards)

    rewards_qq = np.zeros((nQ, nQ))
    for cur_q in range(nQ):
        for event in range(5):
            next_q = transitions[cur_q, event]
            rewards_qq[cur_q, next_q] = rewards[cur_q, event]

    return events, transitions, rewards


class Flower(DiscreteMDP):
    def __init__(self, sizeB, delta, epi_len):  # , ergodic=False):
        self.nO = 6
        self.nA = 2
        self.states = range(0, 6)
        self.actions = range(0, 2)
        self.nameActions = ["A", "M"]

        self.startdistribution = np.zeros((self.nO))
        self.startdistribution[0] = 1.
        self.rewards = {}
        self.P = {}
        self.transitions = {}
        self.epi_len = epi_len

        # Initialize a randomly generated MDP
        for s in self.states:
            self.P[s] = {}
            self.transitions[s] = {}

            # Action A
            self.transitions[s][0] = {}
            self.P[s][0] = []
            li = self.P[s][0]
            if (s == 0):
                for ss in self.states:
                    li.append((1 / 6, ss, False))
                    self.transitions[s][0][ss] = 1 / 6
            else:
                for ss in self.states:
                    if ss == s:
                        li.append((delta, s, False))
                        self.transitions[s][0][ss] = delta
                    elif ss == 0:
                        li.append((1 - delta, 0, False))
                        self.transitions[s][0][ss] = 1 - delta
                    else:
                        li.append((0, s, False))
                        self.transitions[s][0][ss] = 0

            # Action M
            self.transitions[s][1] = {}
            self.P[s][1] = []
            li = self.P[s][1]
            if (s == 0):
                for ss in self.states:
                    li.append((1 / 6, ss, False))
                    self.transitions[s][1][ss] = 1 / 6
            else:
                for ss in self.states:
                    if ss == s:
                        li.append((delta, s, False))
                        self.transitions[s][1][ss] = delta
                    elif ss == 0:
                        li.append((1 - delta, 0, False))
                        self.transitions[s][1][ss] = 1 - delta
                    else:
                        li.append((0, s, False))
                        self.transitions[s][1][ss] = 0

        self.rewards[s] = {}

        # print("Rewards : ", self.rewards, "\nTransitions : ", self.transitions)

        e, t, r = RM_Flower(sizeB)
        self.rewardMachine = RewardMachine2(e, t, r)

        self.Q_star = np.zeros((self.epi_len, self.rewardMachine.n_states, self.nO, self.nA), dtype=np.float64)
        self.V_star = np.zeros((self.epi_len+1, self.rewardMachine.n_states, self.nO), dtype=np.float64)
        self.optimal_policy = np.zeros((self.epi_len, self.rewardMachine.n_states, self.nO), dtype=int)
        self.P_star = np.zeros((self.rewardMachine.n_states, self.nO, self.nA, self.rewardMachine.n_states, self.nO), dtype=np.float64)
        self.optimal_player()

        super(Flower, self).__init__(self.nO, self.nA, self.P, self.rewards, self.startdistribution)

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

    def optimal_player(self):
        # get P and R, using dp to solve for the optimal value function
        nQ = self.rewardMachine.n_states
        R = np.zeros((nQ, self.nO, self.nA))
        # construct real P and R
        for q in range(nQ):
            for o in range(self.nO):
                for a in range(self.nA):
                    # event should be o x a x o
                    event = self.rewardMachine.events[o, a]
                    if event is not None:
                        next_q = self.rewardMachine.transitions[q, event]
                        for z in range(self.nO):
                            #pdb.set_trace()
                            self.P_star[q, o, a, next_q, z] = self.transitions[o][a][z]
                            R[q, o, a] = self.rewardMachine.rewards[q, event]
                    else:
                        next_q = q
                        for z in range(self.nO):
                            self.P_star[q, o, a, next_q, z] = self.transitions[o][a][z]

        # Value Iteration for given P and R
        for h in range(self.epi_len-1, -1, -1):
            for q in range(nQ):
                for o in range(self.nO):
                    for a in range(self.nA):
                        PV = np.sum(self.P_star[q, o, a, :, :]*self.V_star[h+1, :, :])
                        self.Q_star[h, q, o, a] = PV + R[q, o, a]
                    self.optimal_policy[h, q, o] = np.argmax(self.Q_star[h, q, o, :])
                    self.V_star[h, q, o] = np.max(self.Q_star[h, q, o, :])
        print("value iteration finishes.")
if __name__ == "__main__":
    Test = True

    e, t, r = RM_Flower(8)
    print("a")
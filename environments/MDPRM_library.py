import numpy as np
from jinja2 import environment

from utils import *
from environments.discreteMDP import DiscreteMDP
from environments.rewardMachine import RewardMachine, RewardMachine2
from environments.gridWorld import GridWorld, twoRoom, fourRoom
from gym.utils import seeding
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
        self.o = 0
        self.transitions = {}
        self.epi_len = epi_len
        # Initialize a randomly generated MDP
        for o in self.states:
            self.P[o] = {}
            self.transitions[o] = {}
            # GOING RIGHT
            self.transitions[o][0] = {}
            self.P[o][0] = []  # 0=right", 1=left
            li = self.P[o][0]
            prr = 0.
            if (o < self.nO - 1) and (o > 0):
                li.append((rightProbaright, o + 1, False))
                self.transitions[o][0][o + 1] = rightProbaright
                prr = rightProbaright
            elif (o == 0):  # To have 0.6 on the leftmost state
                li.append((0.6, o + 1, False))
                self.transitions[o][0][o + 1] = 0.6
                prr = 0.6
            prl = 0.
            if (o > 0) and (o < self.nO - 1):  # MODIFY HERE FOR THE RIGTHMOST 0.95 and leftmost 0.35
                li.append((rightProbaLeft, o - 1, False))
                self.transitions[o][0][o - 1] = rightProbaLeft
                prl = rightProbaLeft
            elif o == self.nO - 1:  # To have 0.6 and 0.4 on rightmost state
                li.append((0.4, o - 1, False))
                self.transitions[o][0][o - 1] = 0.4
                prl = 0.4
            li.append((1. - prr - prl, o, False))
            self.transitions[o][0][o] = 1. - prr - prl

            # GOING LEFT
            # if ergodic:
            #	pll = 0.95
            # else:
            #	pll = 1
            self.P[o][1] = []  # 0=right", 1=left
            self.transitions[o][1] = {}
            li = self.P[o][1]
            if (o > 0):
                li.append((1., o - 1, False))
                self.transitions[o][1][o - 1] = 1.
            else:
                li.append((1., o, False))
                self.transitions[o][1][o] = 1.

        e, t, r = RM_riverSwim_patrol2(nbStates)
        self.rewardMachine = RewardMachine(e, t, r)
        self.Q_star = np.zeros((self.epi_len, self.rewardMachine.n_states, self.nO, self.nA), dtype=np.float64)
        self.V_star = np.zeros((self.epi_len + 1, self.rewardMachine.n_states, self.nO), dtype=np.float64)
        self.optimal_policy = np.zeros((self.epi_len, self.rewardMachine.n_states, self.nO), dtype=int)
        self.P_star = np.zeros((self.rewardMachine.n_states, self.nO, self.nA, self.rewardMachine.n_states, self.nO),
                               dtype=np.float64)
        self.R = np.zeros((self.rewardMachine.n_states, self.nO, self.nA), dtype=np.float64)

        self.optimal_player()
        self.np_random, _ = seeding.np_random(42)
        # pdb.set_trace()
        # super(RiverSwim_patrol2, self).__init__(self.nO, self.nA, self.P, self.rewards, self.startdistribution)

    def name(self):
        return f"riverswim_{self.nO}o_{self.epi_len}h"

    def reset(self):
        self.rewardMachine.reset()
        self.o = 0
        self.lastaction = None
        return self.o

    def step(self, a):
        event = self.rewardMachine.events[self.o, a]
        r = self.rewardMachine.next_step(event)
        transitions = self.P[self.o][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, o, d = transitions[i]
        # if r == 1:
        #     print("debug")
        self.o = o
        self.lastaction = a
        return (o, r, d, "")

    def optimal_player(self):
        # get P and R, using dp to solve the optimal value function
        nQ = self.rewardMachine.n_states
        # construct real P and R
        for q in range(nQ):
            for o in range(self.nO):
                for a in range(self.nA):
                    # event should be o x a x o
                    event = self.rewardMachine.events[o, a]
                    if event is not None:
                        next_q = self.rewardMachine.transitions[q, event]
                        self.R[q, o, a] = self.rewardMachine.rewards[q, event]
                        for z in self.transitions[o][a].keys():
                            # pdb.set_trace()
                            self.P_star[q, o, a, next_q, z] = self.transitions[o][a][z]
                    else:
                        next_q = q
                        for z in self.transitions[o][a].keys():
                            self.P_star[q, o, a, next_q, z] = self.transitions[o][a][z]

        # Value Iteration for given P and R
        self.Q_star, self.V_star, self.optimal_policy = value_iteration(self.P_star, self.R, self.epi_len)

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

        # self.rewards[s] = {}

        # print("Rewards : ", self.rewards, "\nTransitions : ", self.transitions)

        e, t, r = RM_Flower(sizeB)
        self.rewardMachine = RewardMachine(e, t, r)

        self.Q_star = np.zeros((self.epi_len, self.rewardMachine.n_states, self.nO, self.nA), dtype=np.float64)
        self.V_star = np.zeros((self.epi_len + 1, self.rewardMachine.n_states, self.nO), dtype=np.float64)
        self.optimal_policy = np.zeros((self.epi_len, self.rewardMachine.n_states, self.nO), dtype=int)
        self.P_star = np.zeros((self.rewardMachine.n_states, self.nO, self.nA, self.rewardMachine.n_states, self.nO),
                               dtype=np.float64)
        self.R = np.zeros((self.rewardMachine.n_states, self.nO, self.nA))
        self.optimal_player()

        super(Flower, self).__init__(self.nO, self.nA, self.P, self.rewards, self.startdistribution)

    def name(self):

        return f"flower_{self.rewardMachine.n_states}Q_{self.epi_len}h"

    def reset(self):
        self.rewardMachine.reset()
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        return self.s

    def step(self, a):
        event = self.rewardMachine.events[self.s, a]
        r = self.rewardMachine.next_step(event)
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, d = transitions[i]
        self.s = s
        self.lastaction = a
        return (s, r, d, "")

    def optimal_player(self):
        # get P and R, using dp to solve for the optimal value function
        nQ = self.rewardMachine.n_states
        # construct real P and R
        for q in range(nQ):
            for o in range(self.nO):
                for a in range(self.nA):
                    # event should be o x a x o
                    event = self.rewardMachine.events[o, a]
                    if event is not None:
                        next_q = self.rewardMachine.transitions[q, event]
                        for z in range(self.nO):
                            # pdb.set_trace()
                            self.P_star[q, o, a, next_q, z] = self.transitions[o][a][z]
                            self.R[q, o, a] = self.rewardMachine.rewards[q, event]
                    else:
                        next_q = q
                        for z in range(self.nO):
                            self.P_star[q, o, a, next_q, z] = self.transitions[o][a][z]

        # Value Iteration for given P and R
        self.Q_star, self.V_star, self.optimal_policy = value_iteration(self.P_star, self.R, self.epi_len)


# def mapping_2room(x, y, Y):
#     return y * (Y) + x
#
#
# def RM_twoRoom_4corner(nX, nY):  # 0 = A, 1 = B1, 2 = B2, 3 = C1, 4 = C2, 5 = D1, 6 = D2, 7 = E1, 8 = E2
#     X = nX - 2
#     Y = nY - 2
#     X2 = (int)(X / 2)
#     sA = mapping_2room(X2, 1, Y)
#     sB1 = X2
#     sB2 = 0
#     sC1 = mapping_2room(X2 - 1, 1, Y)
#     sC2 = X - 1
#     sD1 = mapping_2room(X2, 2, Y)
#     sD2 = mapping_2room(0, Y - 1, Y)
#     sE1 = mapping_2room(X2 + 1, 1, Y)
#     sE2 = mapping_2room(X - 1, Y - 1, Y)
#     events = np.array([[None for _ in range(4)] for _ in range(nX * nY)])
#     for a in range(4):
#         events[sA, a] = 0
#         events[sB1, a] = 1
#         events[sB2, a] = 2
#         events[sC1, a] = 3
#         events[sC2, a] = 4
#         events[sD1, a] = 5
#         events[sD2, a] = 6
#         events[sE1, a] = 7
#         events[sE2, a] = 8
#     transitions = np.array([[0, 1, 0, 3, 0, 5, 0, 7, 0],
#                             [1, 1, 2, 1, 1, 1, 1, 1, 1],
#                             [0, 2, 2, 2, 2, 2, 2, 2, 2],
#                             [3, 3, 3, 3, 4, 3, 3, 3, 3],
#                             [0, 4, 4, 4, 4, 4, 4, 4, 4],
#                             [5, 5, 5, 5, 5, 5, 6, 5, 5],
#                             [0, 6, 6, 6, 6, 6, 6, 6, 6],
#                             [7, 7, 7, 7, 7, 7, 7, 7, 8],
#                             [0, 8, 8, 8, 8, 8, 8, 8, 8]])
#     max_i = 9
#     rewards = np.zeros((max_i, max_i))
#     for i in range(max_i):
#         rewards[0, i] = 1
#     return events, transitions, rewards

def to_s(sizeY, rowcol):
        return rowcol[0] * sizeY + rowcol[1]
def RM_tworoom_2corners(sizeX, sizeY):
    # [1, 1]->[sizeX-2, sizeY-2], reward = 1
    nS = sizeX * sizeY
    events = np.array([[None for _ in range(4)] for _ in range(nS)])
    coordinate_A = [1, 1]
    coordinate_B = [sizeX-2, sizeY-2]
    coordinate_center = [(int)(sizeX/2), (int)(sizeY/2)]
    s_A = to_s(sizeY, coordinate_A)
    s_B = to_s(sizeY, coordinate_B)
    s_center = to_s(sizeY, coordinate_center)
    for a in range(4):
        events[s_center, a] = 0
        events[s_A, a] = 1
        events[s_B, a] = 2
    transitions = np.array([[0, 1, 0],
                            [1, 1, 2],
                            [0, 2, 2]])
    rewards = np.array([[0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 0]])
    return events, transitions, rewards
class RM_GridWorld(GridWorld):
    def __init__(self, sizeX, sizeY, epi_len, map_name, slippery=0.1):
        self.sizeX, self.sizeY = sizeX, sizeY
        self.nA = 4
        self.epi_len = epi_len
        # construct maze
        if sizeX <= 3 or sizeY <= 3:
            raise ValueError("Not valid size of grid world, length and width must be greater than 3.")
        if "two_room" in map_name:
            self.maze = twoRoom(sizeX, sizeY)
        elif "four_room" in map_name:
            self.maze = fourRoom(sizeX, sizeY)
        else:
            raise NameError("Invalid map name...")
        print("The maze looks like:\n")
        print(self.maze)
        print("--------------------\n")
        slip = min(1.0 / 3, slippery)
        self.massmap = [[slip, 1. - 3 * slip, slip, 0., slip],  # up : up down left right stay
                        [slip, 0., slip, 1. - 3 * slip, slip],  # down
                        [1. - 3 * slip, slip, 0., slip, slip],  # left
                        [0., slip, 1. - 3 * slip, slip, slip]]  # right

        # extract valid position to be observation space
        self.mapping = []
        for x in range(sizeX):
            for y in range(sizeY):
                if self.maze[x, y] >= 1:
                    self.mapping.append(self.to_s((x, y)))

        # self.nS = len(self.mapping)
        self.nS = sizeX * sizeY
        self.isd = self.makeInitialDistribution(self.maze)
        self.P = self.makeTransition()
        if map_name == "two_room_2corners":
            e, t, r = RM_tworoom_2corners(sizeX, sizeY)

        self.rewardMachine = RewardMachine(e, t, r)
        nQ = self.rewardMachine.n_states
        self.P_star = np.zeros((nQ, self.nS, self.nA, nQ, self.nS))
        self.R = np.zeros((nQ, self.nS, self.nA))
        self.make_PnR()
        self.Q_star = np.zeros((self.epi_len, self.rewardMachine.n_states, self.nS, self.nA), dtype=np.float64)
        self.V_star = np.zeros((self.epi_len + 1, self.rewardMachine.n_states, self.nS), dtype=np.float64)
        self.optimal_policy = np.zeros((self.epi_len, self.rewardMachine.n_states, self.nS), dtype=int)
        self.Q_star, self.V_star, self.optimal_policy = value_iteration(self.P_star, self.R, self.epi_len)
        self.np_random, _ = seeding.np_random(42)
        print("nothing")
    def name(self):
        return f"two_rooms_{self.sizeX}x{self.sizeY}_{self.epi_len}h"
    def make_PnR(self):
        nQ = self.rewardMachine.n_states
        # construct real P and R
        for q in range(nQ):
            # TODO: check if need index() function when index s
            for s in range(self.nS):
                for a in range(self.nA):
                    event = self.rewardMachine.events[s, a]
                    if event is not None:
                        next_q = self.rewardMachine.transitions[q, event]
                        self.R[q, s, a] = self.rewardMachine.rewards[q, event]
                        for (p, next_s, _) in self.P[s][a]:
                            # pdb.set_trace()
                            self.P_star[q, s, a, next_q, next_s] += p
                    else:
                        next_q = q
                        for (p, next_s, _) in self.P[s][a]:
                            self.P_star[q, s, a, next_q, next_s] += p

    def step(self, a):
        event = self.rewardMachine.events[self.s, a]
        r = self.rewardMachine.next_step(event)
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, d = transitions[i]
        self.s = s
        self.lastaction = a
        return (s, r, d, "")

    def reset(self):
        self.rewardMachine.reset()
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        return self.s


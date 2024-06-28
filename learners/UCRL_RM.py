import pdb

import numpy as np
import copy as cp
class UCRL2_RM:
    def __init__(self, nO, nA, epi_len, K, RM, delta):
        self.nO = nO
        self.nA = nA
        self.epi_len = epi_len
        self.K = K
        self.RM = RM
        self.nQ = RM.n_states
        self.t = 1

        self.delta = delta
        self.observations = [[], [], []]
        self.vk = np.zeros((self.nO, self.nA))
        self.Nk = np.zeros((self.nO, self.nA))
        self.Pk = np.zeros((self.nO, self.nA, self.nO))

        self.policy = np.zeros((self.epi_len, self.nQ, self.nO,), dtype=int)
        self.p_distances = np.zeros((self.nO, self.nA))

        # initial state
        self.init_q = RM.init
        self.init_o = 0

        np.random.seed(42)

    def name(self):
        return "UCRL2_RM"

    def reset(self, init_o = 0):
        self.observations = [[init_o], [], []]

    def updateN(self):
        for o in range(self.nO):
            for a in range(self.nA):
                self.Nk[o, a] += self.vk[o, a]

    def updateP(self):
        old_obs = self.observations[0][-2]
        action = self.observations[1][-1]
        new_obs = self.observations[0][-1]

        self.Pk[old_obs, action, new_obs] += 1

    def new_episode(self):
        self.updateN()
        self.vk = np.zeros((self.nO, self.nA))
        p_estimates = np.zeros((self.nO, self.nA, self.nO))

        # get model estimates
        for o in range(self.nO):
            for a in range(self.nA):
                div = max(1, self.Nk[o, a])
                for next_o in range(self.nO):
                    p_estimates[o, a, next_o] += self.Pk[o, a, next_o] / div
                    if np.sum(p_estimates[o, a, :]) > 1+1e-2:
                        print("update rules are wrong")

        self.distances()

        # learn new policy based on new estimated model
        self.EVI(p_estimates)

    def distances(self):
        d = self.delta / (2 * self.nO * self.nA)
        for o in range(self.nO):
            for a in range(self.nA):
                n = max(1, self.Nk[o, a])
                self.p_distances[o, a] = np.sqrt(
                    (2 * (1 + 1 / n) * np.log(np.sqrt(n + 1) * (2 ** (self.nO) - 2) / d)) / n)

    def sorted_indices(self, u0, h, cur_q, cur_o, a):
        u = np.zeros(self.nO)
        event = self.RM.events[cur_o, a]

        if h < self.epi_len - 1:
            if event is not None:
                next_q = self.RM.transitions[cur_q, event]
                u[:] = u0[h + 1, next_q, :] + self.RM.rewards[cur_q, next_q]
            else:
                u[:] = u0[h + 1, cur_q, :]
        else:
            # if h is self.epi_len - 1, next state should always be the starting state
            # TODO: check sanity
            u[:] = u0[0, self.init_q, self.init_o]

        return np.argsort(u)

    def max_proba(self, p_estimate, sorted_indices, o, a):

        min1 = min([1, p_estimate[o, a, sorted_indices[-1]] + (self.p_distances[o, a] / 2)])
        max_p = np.zeros(self.nO)
        if min1 == 1:
            max_p[sorted_indices[-1]] = 1
        else:
            max_p = cp.deepcopy(p_estimate[o, a])
            max_p[sorted_indices[-1]] += self.p_distances[o, a] / 2
            l = 0
            while sum(max_p) > 1:
                max_p[sorted_indices[l]] = max([0, 1 - sum(max_p) + max_p[sorted_indices[l]]])
                l += 1
        return max_p

    def EVI(self, p_estimate, epsilon = 0.01, max_iter = 1000):
        n = 0
        u0 = np.zeros((self.epi_len, self.nQ, self.nO))
        u1 = np.zeros((self.epi_len, self.nQ, self.nO))
        while True:
            n += 1
            stop = True
            for h in range(self.epi_len):
                if h < self.epi_len-1:
                    for q in range(self.nQ):
                        for o in range(self.nO):
                            for a in range(self.nA):
                                max_p = self.max_proba(p_estimate, self.sorted_indices(u0, h, q, o, a), o, a)
                                event = self.RM.events[o, a]
                                if event is not None:
                                    next_q = self.RM.transitions[q, event]
                                    reward = self.RM.rewards[q, next_q]
                                else:
                                    next_q = q
                                    reward = 0
                                temp = reward + np.sum(max_p * u0[h+1, next_q, :])
                                # get argmax a
                                if (a == 0) or (temp > u1[h, q, o]):
                                    #pdb.set_trace()
                                    u1[h, q, o] = temp
                                    self.policy[h, q, o] = a
                else:
                    u1[h, :, :] = u0[0, self.init_q, self.init_o]
                    #u1[h, :, :] = 0
                    self.policy[h, :, :] = 0
                diff = np.abs(u1[h, :, :] - u0[h, :, :])
                if ((np.max(diff) - np.min(diff)) >= epsilon and n < max_iter) or n < 2:
                    stop = False

            u0 = cp.deepcopy(u1)

            if n == max_iter:
                print("No convergence in the EVI")
            if stop is True:
                #print(n)
                #pdb.set_trace()
                break

    def play(self, h, cur_q, cur_o):
        action = self.policy[h, cur_q, cur_o]
        if self.vk[cur_o, action] >= max(1, self.Nk[cur_o, action]):
            self.new_episode()
            action = self.policy[h, cur_q, cur_o]
        return action

    def update(self, cur_q, action, reward, observation, timestep):
        if timestep < self.epi_len - 1:
            self.vk[self.observations[0][-1], action] += 1
            self.observations[0].append(observation)
            self.observations[1].append(action)
            self.observations[2].append(reward)
            self.updateP()
        self.t += 1
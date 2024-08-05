import numpy as np
from utils import calculate_variance
import pdb
import math


class UCBVI_PRM:
    def __init__(self, nO, nA, epi_len, delta, K, RM, bonus_scale=0.01):
        self.nO = nO
        self.nA = nA
        self.epi_len = epi_len
        self.p = np.zeros((nO, nA, nO))
        self.delta = delta
        self.K = K
        self.bonus_scale = bonus_scale
        self.N_oaz = np.zeros((nO, nA, nO))
        self.N = np.zeros((nO, nA))
        self.N_h = np.zeros((epi_len, nO))
        self.observations_buffer = [[], [], [], []]
        self.RM = RM
        self.nQ = RM.n_states
        self.Q = np.ones((epi_len, self.nQ, nO, nA))*self.epi_len
        self.P = np.zeros((self.nQ, nO, nA, self.nQ, nO))
        self.R = np.zeros((self.nQ, nO, nA))
        self.policy = np.zeros((self.epi_len, self.nQ, self.nO,), dtype=int)
        self.doubling_trick = True
        #np.random.seed(42)
    def name(self):
        return 'UCBVI_PRM'

    def reset(self, initial_obs):
        self.observations_buffer = [[initial_obs], [], [], [0]]

    def update(self, rm_state, action, reward, observation, time):

        self.observations_buffer[0].append(observation)
        self.observations_buffer[1].append(action)
        self.observations_buffer[2].append(reward)
        self.observations_buffer[3].append(time)
        self.update_N()

    def update_N(self):
        self.N_oaz[self.observations_buffer[0][-2], self.observations_buffer[1][-1], self.observations_buffer[0][-1]] += 1
        self.N_h[self.observations_buffer[3][-1], self.observations_buffer[0][-1]] += 1
        self.N[self.observations_buffer[0][-2], self.observations_buffer[1][-1]] += 1
        if self.doubling_trick:
            if math.log2(self.N[self.observations_buffer[0][-2], self.observations_buffer[1][-1]]).is_integer():
                self.update_transition_prob_obs()
                self.update_transition_prob_state()
                self.update_rewards()
                self.update_Q()

    def learn(self):
        if self.doubling_trick is False:
            self.update_transition_prob_obs()
            self.update_transition_prob_state()
            self.update_rewards()
            self.update_Q()
        else:
            pass

    def bonus(self, h, q, o, a, V):
        # T = self.K * self.epi_len
        # L = np.log(6*self.nO*self.nA*T/self.delta)
        # var_W = self.calculate_var_W(V, h+1, q, o, a)


        T = self.K * self.epi_len
        L = np.log(6 * self.nO * self.nA * T / self.delta)
        var_W = self.calculate_var_W(V, h + 1, q, o, a)

        # Precompute some repetitive terms
        sqrt_L = np.sqrt(L)
        sqrt_8L = np.sqrt(8 * L)
        epi_len_sq = self.epi_len ** 2
        epi_len_cube = self.epi_len ** 3
        const_regret = 10000 * epi_len_cube * (self.nO ** 2) * self.nA * (L ** 2)

        temp = 0.0
        if h < self.epi_len - 1:
            valid_indices = self.N_h[h + 1, :] > 0
            regret_state = np.ones(self.nO)*epi_len_sq
            regret_state[valid_indices] = const_regret / self.N_h[h + 1, valid_indices]
            regret_state = np.minimum(epi_len_sq, regret_state)
            temp += np.dot(self.p[o, a, :], regret_state)
            # temp += np.dot(self.p[o, a, ~valid_indices], epi_len_sq)
        else:
            temp += epi_len_sq
        N_oa = self.N[o, a]
        bonus = (
                sqrt_8L * np.sqrt(var_W / N_oa)
                + 14 * self.epi_len * L / (3 * N_oa)
                + np.sqrt(8 * temp / N_oa)
                + np.sqrt(2 * L / N_oa)
        )
        # bonus = np.sqrt(8*L*var_W/self.N[o, a]) + 14*self.epi_len*L/(3*self.N[o, a]) + np.sqrt(8*temp/self.N[o, a]) + np.sqrt(2*L/self.N[o, a])
        return bonus

    def calculate_var_W(self, V, h, q, o, a):
        # calculate W_h
        W_h = np.zeros(self.nO, dtype=np.float64)
        for z in range(self.nO):
            event = self.RM.events[o, a, z]
            if event is not None:
                for next_q in range(self.nQ):
                    W_h[z] += self.RM.transitions[q, event, next_q]*V[h, next_q, z]
            else:
                # not defined event, q stays the same
                W_h[z] += V[h, q, z]
        # to do: check compatibility in dimensions
        var_W = calculate_variance(self.p[o, a, :], W_h)
        return var_W


    def update_transition_prob_obs(self):
        for o in range(self.nO):
            for a in range(self.nA):
                if self.N[o, a] > 0:
                    self.p[o, a, :] = self.N_oaz[o, a, :] / (self.N[o, a])

    def update_transition_prob_state(self):
        for q in range(self.nQ):
            for o in range(self.nO):
                for a in range(self.nA):
                    for z in range(self.nO):
                        if self.p[o, a, z] >= 0:
                            event = self.RM.events[o, a, z]
                            if event is not None:
                                for next_q in range(self.nQ):
                                    if self.RM.transitions[q, event, next_q] > 0:
                                        self.P[q, o, a, next_q, z] = (self.p[o, a, z] *
                                                                      self.RM.transitions[q, event, next_q])
                            else:
                                self.P[q, o, a, q, z] = self.p[o, a, z]

    def update_rewards(self):
        for q in range(self.nQ):
            for o in range(self.nO):
                for a in range(self.nA):
                    #self.R[q, o, a] = np.sum(self.P[q, o, a, :, :] * (self.RM.rewards[q, :].reshape(self.nQ, 1)))
                    temp = 0.0
                    for z in range(self.nO):
                        event = self.RM.events[o, a, z]
                        if event is not None:
                            for next_q in range(self.nQ):
                                temp += self.P[q, o, a, next_q, z] * self.RM.rewards[q, event, next_q]

                    self.R[q, o, a] = temp
                    #if self.R[q, o, a] != np.sum(self.P[q, o, a, :, :] * (self.RM.rewards[q, :].reshape(self.nQ, 1))):
                    #    pdb.set_trace()


    def update_Q(self):
        V = np.zeros((self.epi_len+1, self.nQ, self.nO))
        # V_{H+1} = 0 for all s
        for h in range(self.epi_len-1, -1, -1):
            for q in range(self.nQ):
                for o in range(self.nO):
                    for a in range(self.nA):
                        if self.N[o, a] > 0:
                            bonus = self.bonus(h, q, o, a, V)*self.bonus_scale
                            # print("Time for bonus: ", time.time() - str_time)
                            PV = np.sum(self.P[q, o, a, :, :] * V[h+1, :, :])
                            self.Q[h, q, o, a] = min(min(self.Q[h, q, o, a], self.epi_len), self.R[q, o, a] + PV + bonus)
                        else:
                            self.Q[h, q, o, a] = self.epi_len
                    V[h, q, o] = np.max(self.Q[h, q, o, :])

    def play(self, h, q, o):
        max_Q = np.max(self.Q[h, q, o, :])
        max_actions = np.where(self.Q[h, q, o, :] == max_Q)

        action = np.random.choice(max_actions[0])
        self.policy[h, q, o] = action
        # self.policy[h, q, o] = np.argmax(self.Q[h, q, o, :])

        return self.policy[h, q, o]

    def get_policy(self):
        return self.policy


class UCBVI_RM(UCBVI_PRM):
    def __init__(self, nO, nA, epi_len, delta, K, RM, bonus_scale=0.01):
        super(UCBVI_RM, self).__init__(nO, nA, epi_len, delta, K, RM, bonus_scale)

    def name(self):
        return 'UCBVI_RM'
    def update_transition_prob_state(self):
        for q in range(self.nQ):
            for o in range(self.nO):
                for a in range(self.nA):
                    # event should be o x a x o
                    event = self.RM.events[o, a]
                    if event is not None:
                        next_q = self.RM.transitions[q, event]
                        for z in range(self.nO):
                            #pdb.set_trace()
                            self.P[q, o, a, next_q, z] = self.p[o, a, z]
                    else:
                        next_q = q
                        for z in range(self.nO):
                            self.P[q, o, a, next_q, z] = self.p[o, a, z]

    def calculate_var_W(self, V, h, q, o, a):
        # calculate W_h
        W_h = np.zeros(self.nO)

        for z in range(self.nO):
            # pdb.set_trace()
            event = self.RM.events[o, a]
            if event is not None:
                next_q = self.RM.transitions[q, event]
                W_h[z] += V[h, next_q, z]
            else:
                # not defined event, q stays the same
                W_h[z] += V[h, q, z]

        var_W = calculate_variance(self.p[o, a, :], W_h)
        return var_W

    def update_rewards(self):
        for q in range(self.nQ):
            for o in range(self.nO):
                for a in range(self.nA):
                    #self.R[q, o, a] = np.sum(self.P[q, o, a, :, :] * (self.RM.rewards[q, :].reshape(self.nQ, 1)))
                    event = self.RM.events[o, a]
                    if event is not None:
                        self.R[q, o, a] = self.RM.rewards[q, event]
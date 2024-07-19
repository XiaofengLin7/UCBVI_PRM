import numpy as np
from utils import calculate_variance
import math

class UCBVI_CP():
    def __init__(self, nQ, nO, nA, epi_len, delta, K, rm_rewards):
        self.nQ = nQ
        self.nO = nO
        self.nA = nA
        self.epi_len = epi_len
        self.delta = delta
        self.K = K

        self.Q = np.zeros((epi_len, self.nQ, nO, nA))
        self.P = np.zeros((self.nQ, nO, nA, self.nQ, nO))
        self.R = np.zeros((self.nQ, nO, nA))

        self.N_xay = np.zeros((nQ, nO, nA, nQ, nO))
        self.N = np.zeros((nQ, nO, nA))
        self.N_h = np.zeros((epi_len, nQ, nO))
        self.observations_buffer = [[], [], [], [], []]

        self.initial_q = 0
        self.rm_rewards = rm_rewards
        self.doubling_trick = True
        np.random.seed(42)
    def name(self):
        return "UCBVI_CP"

    def reset(self, initial_obs):
        self.observations_buffer = [[self.initial_q], [initial_obs],[],[],[0]]

    def update(self, rm_state, action, reward, observation, time):

        self.observations_buffer[0].append(rm_state)
        self.observations_buffer[1].append(observation)
        self.observations_buffer[2].append(action)
        self.observations_buffer[3].append(reward)
        self.observations_buffer[4].append(time)
        self.update_N()

    def update_N(self):
        old_q = self.observations_buffer[0][-2]
        new_q = self.observations_buffer[0][-1]
        old_obs = self.observations_buffer[1][-2]
        new_obs = self.observations_buffer[1][-1]
        action = self.observations_buffer[2][-1]
        time = self.observations_buffer[4][-1]

        self.N_xay[old_q, old_obs, action, new_q, new_obs] += 1
        self.N_h[time, new_q, new_obs] += 1
        self.N[old_q, old_obs, action] += 1
        if self.doubling_trick:
            if math.log2(self.N[old_q, old_obs, action]).is_integer():
                self.update_transition_prob()
                self.update_rewards()
                self.update_Q()


    def update_transition_prob(self):
        for q in range(self.nQ):
            for obs in range(self.nO):
                for a in range(self.nA):
                    if self.N[q, obs, a] > 0:
                        self.P[q, obs, a, :, :] = self.N_xay[q, obs, a, :, :] / self.N[q, obs, a]

    def update_rewards(self):
        for q in range(self.nQ):
            for o in range(self.nO):
                for a in range(self.nA):
                    self.R[q, o, a] = 0.0
                    for z in range(self.nO):
                        for next_q in range(self.nQ):
                            self.R[q, o, a] += self.P[q, o, a, next_q, z] * np.sum(self.rm_rewards[q, :, next_q])

    def bonus(self, h, q, o, a, V) -> float:
        T = self.K * self.epi_len
        L = np.log(5*self.nO*self.nA*T/self.delta)
        # check dimensions
        var_V = calculate_variance(self.P[q, o, a, :, :].reshape((self.nQ*self.nO, 1)),
                                   V[h+1, :, :].reshape((self.nQ*self.nO, 1)))
        temp = 0.0
        for next_q in range(self.nQ):
            for next_o in range(self.nO):
                if (h < self.epi_len-1) and (self.N_h[h+1, next_q, next_o]) > 0:
                    regret_state = (10000 * (self.epi_len**3) * (self.nO**2) * (self.nQ**2) * self.nA * (L**2) /
                                    self.N_h[h+1, next_q, next_o])
                    temp += self.P[q, o, a, next_q, next_o] * min(self.epi_len**2, regret_state)
                else:
                    temp += self.P[q, o, a, next_q, next_o] * self.epi_len**2

        bonus = (np.sqrt(8*L*var_V/self.N[q, o, a]) + 14*self.epi_len*L/(3*self.N[q, o, a]) +
                 np.sqrt(8*temp/self.N[q, o, a]))
        return bonus

    def update_Q(self):
        V = np.zeros((self.epi_len+1, self.nQ, self.nO))
        # V_H = 0 for all s
        for h in range(self.epi_len-1, -1, -1):
            for q in range(self.nQ):
                for o in range(self.nO):
                    for a in range(self.nA):
                        if self.N[q, o, a] > 0:
                            PV = np.sum(self.P[q, o, a, :, :]*V[h+1, :, :])
                            bonus = self.bonus(h, q, o, a, V)
                            self.Q[h, q, o, a] = min(min(self.Q[h, q, o, a], self.epi_len),
                                                     self.R[q, o, a] + PV + bonus)
                        else:
                            self.Q[h, q, o, a] = self.epi_len

                    V[h, q, o] = np.max(self.Q[h, q, o, :])

    def learn(self):
        if self.doubling_trick is False:
            self.update_transition_prob()
            self.update_rewards()
            self.update_Q()
        else:
            pass # already updated in update_N

    def play(self, h, q, o):
        max_Q = np.max(self.Q[h, q, o, :])
        max_actions = np.where(self.Q[h, q, o, :] == max_Q)

        action = np.random.choice(max_actions[0])

        return action
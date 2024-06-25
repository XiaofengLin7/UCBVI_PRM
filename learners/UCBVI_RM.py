import numpy as np
from utils import calculate_variance
import pdb
class UCBVI_PRM:
    def __init__(self, nO, nA, epi_len, delta, K, PRM):
        self.nO = nO
        self.nA = nA
        self.epi_len = epi_len
        self.p = np.zeros((nO, nA, nO))
        self.delta = delta
        self.K = K

        self.N_oaz = np.zeros((nO, nA, nO))
        self.N = np.zeros((nO, nA))
        self.N_h = np.zeros((epi_len, nO))
        self.observations_buffer = [[], [], [], []]
        self.RM = PRM
        self.nQ = PRM.n_states
        self.Q = np.zeros((epi_len, self.nQ, nO, nA))
        self.P = np.zeros((self.nQ, nO, nA, self.nQ, nO))
        self.R = np.zeros((self.nQ, nO, nA))

    def name(self):
        return 'UCBVI_PRM'

    def reset(self, initial_obs):
        self.observations_buffer = [[initial_obs],[],[],[0]]

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

    def learn(self):
        self.update_transition_prob_obs()
        self.update_transition_prob_state()
        self.update_rewards()
        self.update_Q()

    def bonus(self, h, q, o, a, V):
        T = self.K * self.epi_len
        L = np.log(6*self.nQ*self.nO*self.nA*T/self.delta)
        var_W = self.calculate_var_W(V, h+1, q, o, a)
        temp = 0.0
        for z in range(self.nO):
            if self.N_h[h+1, z] > 0:
                regret_state = 10000 * (self.epi_len**3) * (self.nO**2) * self.nA * (L**2) / self.N_h[h+1, z]
                temp += self.p[o, a, z] * min(self.epi_len**2, regret_state)
            else:
                temp += self.p[o, a, z] * self.epi_len**2

        bonus = np.sqrt(8*L*var_W/self.N[o, a]) + np.sqrt(2*L/self.N[o, a]) + 14*self.epi_len/(3*self.N[o, a]) + np.sqrt(8*temp/self.N[o, a])
        return bonus

    def calculate_var_W(self, V, h, q, o, a):
        # calculate W_h
        W_h = np.zeros(self.nO)
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

    def calculate_PV(self, V, h, q, o, a):
        # return PV_h(s, a) = \sum_{s'} P(s'|s, a) V_h(s')
        ret = 0.0
        for next_q in range(self.nQ):
            for next_o in range(self.nO):
                ret += self.P[q, o, a, next_q, next_o]*V[h, next_q, next_o]

        return ret

    def update_transition_prob_obs(self):
        for o in range(self.nO):
            for a in range(self.nA):
                if self.N[o, a] > 0:
                    for z in range(self.nO):
                        #pdb.set_trace()
                        self.p[o, a, z] = self.N_oaz[o, a, z] / (self.N[o, a])

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
                                        self.P[q, o, a, z, next_q] = (self.p[o, a, z] *
                                                                      self.RM.transitions[q, event, next_q])
                            else:
                                self.P[q, o, a, z, q] = self.p[o, a, z]

    def update_rewards(self):
        for q in range(self.nQ):
            for o in range(self.nO):
                for a in range(self.nA):
                    self.R[q, o, a] = 0.0
                    for z in range(self.nO):
                        for next_q in range(self.nQ):
                            self.R[q, o, a] += self.P[q, o, a, next_q, z]*self.RM.rewards[q, next_q]


    def update_Q(self):
        V = np.zeros((self.epi_len, self.nQ, self.nO))
        # V_H = 0 for all s
        for h in range(self.epi_len-2, -1, -1):
            for q in range(self.nQ):
                for o in range(self.nO):
                    for a in range(self.nA):
                        if self.N[o, a] > 0:
                            bonus = self.bonus(h, q, o, a, V)
                            PV = self.calculate_PV(V, h+1, q, o, a)
                            self.Q[h, q, o, a] = min(min(self.Q[h, q, o, a], self.epi_len), self.R[q, o, a] + PV + bonus)
                        else:
                            self.Q[h, q, o, a] = self.epi_len
                    V[h, q, o] = max(self.Q[h, q, o, a] for a in range(self.nA))

    def play(self, h, q, o):
        max_Q = np.max(self.Q[h, q, o, :])
        max_actions = np.where(self.Q[h, q, o, :] == max_Q)

        action = np.random.choice(max_actions[0])

        return action


class UCBVI_RM(UCBVI_PRM):
    def __init__(self, nO, nA, epi_len, delta, K, RM):
        super(UCBVI_RM, self).__init__(nO, nA, epi_len, delta, K, RM)

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

        # to do: check compatibility in dimensions
        var_W = calculate_variance(self.p[o, a, :], W_h)
        return var_W
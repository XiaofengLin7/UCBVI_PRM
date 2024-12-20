from learners.UCRL_RM import UCRL2_RM
import numpy as np
class UCRL2_RM_Bernstein(UCRL2_RM):
    def __init__(self, nO, nA, epi_len, K, RM, delta, distance_scale):
        self.nO = nO
        self.nA = nA
        self.epi_len = epi_len
        self.K = K
        self.RM = RM
        self.nQ = RM.n_states
        self.t = 1
        self.distance_scale = distance_scale
        self.delta = delta
        self.observations = [[], [], []]
        self.vk = np.zeros((self.nO, self.nA))
        self.Nk = np.zeros((self.nO, self.nA))
        self.Pk = np.zeros((self.nO, self.nA, self.nO))

        self.policy = np.zeros((self.epi_len, self.nQ, self.nO,), dtype=int)
        self.p_distances = np.zeros((self.nO, self.nA, self.nO, 2))

        # initial state
        self.init_q = RM.init
        self.init_o = 0
        self.params = distance_scale
    def name(self):
        return "UCRL2-RM-B"

    def beta(self, n, delta):
        eta = 1.12
        temp = eta * np.log(np.log(max((np.exp(1), n))) * np.log(eta * max(np.exp(1), n)) / (delta * np.log(eta) ** 2))
        return temp

    def upper_bound(self, n, p_est, bound_max, beta):
        up = min(p_est + bound_max*self.distance_scale, 1)
        down = p_est
        for _ in range(5):
            m = (up + down) / 2
            # m = min(1, m)
            temp = self.distance_scale*(np.sqrt(2 * beta * m * (1 - m) / n) + beta / (3 * n))
            if m - temp <= p_est:
                down = m
            else:
                up = m
        return (up + down) / 2

    def lower_bound(self, n, p_est, bound_max, beta):
        down = max(p_est - bound_max*self.distance_scale, 0)
        up = p_est
        for _ in range(5):
            m = (up + down) / 2
            # m = max(0, m)
            temp = self.distance_scale*(np.sqrt(2 * beta * m * (1 - m) / n) + beta / (3 * n))
            if m + temp >= p_est:
                up = m
            else:
                down = m
        return (up + down) / 2

    def distances(self, p_estimate):
        delta = self.delta / (2 * self.nO * self.nA)
        for s in range(self.nO):
            for a in range(self.nA):
                n = max(1, self.Nk[s, a])
                for next_s in range(self.nO):
                    p = p_estimate[s, a, next_s]
                    beta = self.beta(n, delta)
                    bound_max = np.sqrt(beta / (2 * n)) + beta / (3 * n)
                    lower_bound = self.lower_bound(n, p, bound_max, beta)
                    upper_bound = self.upper_bound(n, p, bound_max, beta)
                    self.p_distances[s, a, next_s, 0] = lower_bound
                    self.p_distances[s, a, next_s, 1] = upper_bound

    def max_proba(self, p_estimate, sorted_indices, s, a, epsilon=10 ** (-8), reverse=False):
        max_p = np.zeros(self.nO)
        delta = 1.
        for next_s in range(self.nO):
            max_p[next_s] = max((0, p_estimate[s, a, next_s] - self.p_distances[s, a, next_s, 0]))
            delta += - max_p[next_s]
        l = 0
        while (delta > 0) and (l <= self.nO - 1):
            idx = self.nO - 1 - l if not reverse else l
            idx = sorted_indices[idx]
            new_delta = min((delta, p_estimate[s, a, idx] + self.p_distances[s, a, idx, 1] - max_p[idx]))
            max_p[idx] += new_delta
            delta += - new_delta
            l += 1
        return max_p
    def new_episode(self):
        self.updateN()  # Don't run it after the reinitialization of self.vk
        self.vk = np.zeros((self.nO, self.nA))
        p_estimate = np.zeros((self.nO, self.nA, self.nO))
        for s in range(self.nO):
            for a in range(self.nA):
                div = max([1, self.Nk[s, a]])
                for next_s in range(self.nO):
                    p_estimate[s, a, next_s] = self.Pk[s, a, next_s] / div
        self.distances(p_estimate)
        self.EVI(p_estimate)
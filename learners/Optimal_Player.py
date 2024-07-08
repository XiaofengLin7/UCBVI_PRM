import numpy as np

class Optimal_Player:
    def __init__(self, env, K):
        #self.P_star = env.P_star
        # self.policy = env.optimal_policy
        self.Q_star = env.Q_star
        self.K = K
    def name(self):
        return "Optimal"
    def reset(self, init_o):
        ()

    def update(self, cur_q, action, reward, observation, timestep):
        ()

    def play(self, h, cur_q, cur_o):
        action_value = self.Q_star[h, cur_q, cur_o, :]
        action = np.random.choice(np.where(action_value == action_value.max())[0])
        return action
    def learn(self):
        pass

import numpy as np
from gym import Env, spaces
from gym.utils import seeding
from gym import utils
import string

from utils import categorical_sample, clip
class DiscreteMDP(Env):
    def __init__(self, nS, nA, P, R, isd, nameActions=[]):
        self.P = P
        self.R = R
        self.isd = isd
        self.lastaction = None  # for rendering
        self.nS = nS
        self.nA = nA
        self.render_mode = 'None'

        self.states = range(0, self.nS)
        self.actions = range(0, self.nA)
        if (len(nameActions) == 0):
            self.nameActions = list(string.ascii_uppercase)[0:min(self.nA, 26)]

        self.reward_range = (0, 1)
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.initializedRender = False
        self.seed(42)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        return self.s

    def step(self, a):
        transitions = self.P[self.s][a]
        rewarddis = self.R[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, d = transitions[i]
        r = clip(rewarddis.rvs(), self.reward_range)
        self.s = s
        self.lastaction = a
        return (s, r, d, "")

    def render(self):
        if self.render_mode == 'None':
            print('Do nothing')
        # TO DO: render

    def getTransition(self, s, a):
        transition = np.zeros(self.nS)
        for c in self.P[s][a]:
            transition[c[1]] = c[0]
        return transition

    # nb_iter is the number of reward samples used to compute the mean of the reward for the given pair of state-action.
    def getReward(self, s, a, nb_iter=1):
        rewarddis = self.R[s][a]
        r = np.mean([clip(rewarddis.rvs(), self.reward_range) for _ in range(nb_iter)])
        return r


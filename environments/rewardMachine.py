from utils import categorical_sample

# A reward machine is defined by a number of states (states inditified by int),  a list of high level events (we chose to use
# use the number of events and identify them as int), a list of transtions between states (matrix statexevent=next_state) and a
# list of rewards (matrix statexstate=rewards) that can be given when a transtion occurs.
class RewardMachine:
    def __init__(self, Events, Transitions, Rewards, init = 0):
        self.n_states = len(Transitions)
        self.events = Events # SxAxS matrix with corresponding event or None
        self.transitions = Transitions
        self.rewards = Rewards
        self.current_state = init
        self.previous_state = init
        self.init = init

    def next_step(self, event):
        reward = 0
        self.previous_state = self.current_state
        if event is not None:
            old_state = self.current_state
            self.current_state = self.transitions[self.current_state, event]
            reward = self.rewards[old_state, event]
            # if reward == 1:
            #     print("stop")
        return reward

    def reset(self):
        self.previous_state = self.current_state
        self.current_state = self.init
class RewardMachine2(RewardMachine):
    def __init__(self, Events, Transitions, Rewards, init=0, np_random = None):
        super().__init__(Events, Transitions, Rewards, init)
        self.np_random = np_random

    def next_step(self, event):
        reward = 0
        self.previous_state = self.current_state
        if event is not None:
            old_state = self.current_state
            self.current_state = self.transitions[self.current_state, event]
            reward = self.rewards[old_state, event]
        return reward

class ProbabilisticRewardMachine(RewardMachine):
    def __init__(self, Events, Transitions, Rewards, init = 0, np_random = None):
        super().__init__(Events, Transitions, Rewards, init)
        self.np_random = np_random
    def next_step(self, event):
        reward = 0
        self.previous_state = self.current_state
        if event is not None:
            old_state = self.current_state
            self.current_state = categorical_sample(self.transitions[old_state, event, :], self.np_random)
            reward = self.rewards[old_state, event, self.current_state]
        return reward

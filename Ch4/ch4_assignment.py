from abc import ABC, abstractmethod
import numpy as np

class Game(ABC):
    states = []
    actions = {}

    @abstractmethod
    def get_state_and_reward(self, oldstate, action):
        pass

    def make_equiprobable_random_policy(self):
        policy = {}
        for st in self.states:
            policy[st] = {}
            n_act = len(self.actions[st])
            for act in self.actions[st]:
                policy[st][act] = 1 / n_act
        return policy

class GridWorld(Game):

    def __init__(self):
        self.states = list(range(15))
        self.actions = {st : ["U", "D", "L", "R"] for st in self.states}
        self.actions[0] = []

    def get_state_and_reward(self, oldstate, action):
        if action not in self.actions[oldstate]:
            raise Exception("Impossible action.")
        
        if action == "U":
            if oldstate >= 4:
                return [((oldstate - 4) % 15, -1)], [1]
        
        if action == "D":
            if oldstate <= 11:
                return [((oldstate + 4) % 15, -1)], [1]
        
        if action == "R":
            if oldstate % 4 <= 2:
                return [((oldstate + 1) % 15, -1)], [1]
        
        if action == "L":
            if oldstate % 4 >= 1:
                return [((oldstate - 1) % 15, -1)], [1]
            
        return [(oldstate, -1)], [1]
        
class DynamicProgram:
    game = None

    def __init__(self, game):
        self.game = game

    def policy_evaluation(self, policy, tol=1.0, disc=1.0):
        value = {st: 0 for st in self.game.states}
        delta = tol + 1
        counter = 0
        while delta > tol:
            delta = 0
            old_value = value.copy()
            for st in self.game.states:
                value[st] = 0
                for act in self.game.actions[st]:
                    prob_act = policy[st][act]
                    state_reward_list, prob_state_reward_list = self.game.get_state_and_reward(st, act)
                    for state_reward, prob_state_reward in zip(state_reward_list, prob_state_reward_list):
                        # print(prob_act, prob_state_reward, state_reward[1], st, state_reward[0])
                        value[st] += prob_act * prob_state_reward * (state_reward[1] + disc * old_value[state_reward[0]])
                delta = max(delta, abs(old_value[st] - value[st]))
            counter += 1
        return value

gw = GridWorld()
dp = DynamicProgram(gw)
erp = gw.make_equiprobable_random_policy()
value = dp.policy_evaluation(erp, tol=0)
print(value)
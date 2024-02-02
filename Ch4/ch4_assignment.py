from abc import ABC, abstractmethod
import numpy as np

class Game(ABC):
    '''
    Abstract interface for general games
    '''
    states = []     # list of states
    actions = {}    # dictionary mapping each state to a list of possible actions

    @abstractmethod
    def get_state_and_reward(self, oldstate, action):
        '''
        Given the current state and an action, outputs two parallel lists describing potential outcomes
        RETURNS
            list of 2-tuples, where the first tuple element is the potential next state and second tuple element is the potential reward
            list of probabilities of achieving the associated state and reward
        '''
        pass

    def make_equiprobable_random_policy(self):
        '''
        RETURNS equiprobable random policy as a dictionary
        maps each state to a dictionary, which maps each action to the probability of selecting that action
        '''
        policy = {}
        for st in self.states:
            policy[st] = {}

            # Compute equiprobable random policy
            n_act = len(self.actions[st])

            # Assign probabilities for actions
            for act in self.actions[st]:
                policy[st][act] = 1 / n_act
        return policy

class GridWorld(Game):
    '''
    Implements the GridWorld game
    '''

    def __init__(self):
        '''
        Initializes this object to contain states 1-14 with absorbing state 0
        Intiializes UDLR actions for all states except absorbing state 0 
        '''
        self.states = list(range(15))
        self.actions = {st : ["U", "D", "L", "R"] for st in self.states}
        self.actions[0] = []

    def get_state_and_reward(self, oldstate, action):
        '''
        Given the current state and an action, returns two parallel lists describing the outcome
        For GridWorld, the lists are all length 1 because the actions all have deterministic outcome
        RETURNS
            list of 2-tuples, where the first tuple element is the potential next state and second tuple element is the potential reward
            list of probabilities of achieving the associated state and reward
        '''
        if action not in self.actions[oldstate]:
            raise Exception("Impossible action.")
        
        # Check if UP is valid move
        if action == "U":
            if oldstate >= 4:
                return [((oldstate - 4) % 15, -1)], [1]
        
        # Check if DOWN is valid move
        if action == "D":
            if oldstate <= 11:
                return [((oldstate + 4) % 15, -1)], [1]
        
        # Check if RIGHT is valid move
        if action == "R":
            if oldstate % 4 <= 2:
                return [((oldstate + 1) % 15, -1)], [1]
        
        # Check if LEFT is valid move
        if action == "L":
            if oldstate % 4 >= 1:
                return [((oldstate - 1) % 15, -1)], [1]
            
        # If move would go off board, then stay where you are
        return [(oldstate, -1)], [1]
        
class DynamicProgram:
    '''
    Implements policy evaluation, policy iteration, and value iteration.
    '''
    game = None     # the game associated with this Dynamic Program object

    def __init__(self, game):
        '''
        Initializes this object
        INPUT
            game; the associated game object
        '''
        self.game = game

    def policy_evaluation(self, policy, tol=1.0, disc=1.0):
        '''
        Executes policy evaluation
        INPUT
            policy; dictionary mapping each state to a dictionary that maps each action to the probability of selecting that action
            tol; optimality tolerance
            disc; discount rate
        RETURNS
            disctionary mapping each state to its value function under given policy
        '''
        # Initialize value function
        value = {st: 0 for st in self.game.states}
        
        # Set delta to high number
        delta = tol + 1

        # While change in value function is high, loop
        while delta > tol:
            delta = 0

            # Store old values for computing change in values this iteration
            old_value = value.copy()

            # Compute new values for each state
            for st in self.game.states:

                value[st] = 0
                
                # Sum over actions
                for act in self.game.actions[st]:
                    prob_act = policy[st][act]  # probability of selecting the action given the current state

                    # Get potential action outcomes
                    state_reward_list, prob_state_reward_list = self.game.get_state_and_reward(st, act)
                    for state_reward, prob_state_reward in zip(state_reward_list, prob_state_reward_list):

                        # Compute contributino to the value function
                        value[st] += prob_act * prob_state_reward * (state_reward[1] + disc * old_value[state_reward[0]])
                
                # Compute change in this state's value function
                delta = max(delta, abs(old_value[st] - value[st]))
        return value    

gw = GridWorld()
dp = DynamicProgram(gw)
erp = gw.make_equiprobable_random_policy()
value = dp.policy_evaluation(erp, tol=0)
print(value)
from abc import ABC, abstractmethod
import random
import numpy as np
import structlog
from typing import List, Tuple

logger = structlog.get_logger()

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
    game = None     # Game associated with this Dynamic Program object
    gamma = 1.0     # Discount rate
    tol = 0.0       # Optimality tolerance

    def __init__(self, game, gamma=1.0, tol=0.0):
        '''
        Initializes this object
        INPUT
            game; the associated game object
            gamma; discount rate between 0 and 1; 1 implies undiscounted model
            tol; optimality tolerance
        '''
        self.game: Game = game
        self.gamma = gamma
        self.tol = tol

    def policy_evaluation(self, policy):
        '''
        Executes policy evaluation
        INPUT
            policy; dictionary mapping each state to a dictionary that maps each action to the probability of selecting that action
        RETURNS
            disctionary mapping each state to its value function under given policy
        '''
        # Initialize value function
        value = {st: 0 for st in self.game.states}
        
        # Set delta to high number
        delta = self.tol + 1

        # While change in value function is high, loop
        while delta > self.tol:
            delta = 0

            # Store old values for computing change in values this iteration
            old_value = value.copy()

            # Compute new values for each state
            for st in self.game.states:

                value[st] = 0
                
                # Sum over actions
                for act in policy[st].keys():
                    prob_act = policy[st][act]  # probability of selecting the action given the current state

                    # Get potential action outcomes
                    state_reward_list, prob_state_reward_list = self.game.get_state_and_reward(st, act)
                    for state_reward, prob_state_reward in zip(state_reward_list, prob_state_reward_list):

                        # Compute contribution to the value function
                        value[st] += prob_act * prob_state_reward * (state_reward[1] + self.gamma * old_value[state_reward[0]])
                
                # Compute change in this state's value function
                delta = max(delta, abs(old_value[st] - value[st]))
        return value   

    def policy_improvement(self, policy, value, disc=1.0):
        '''
        Mutates and improves the given policy
        Assumes the given policy is deterministic: each state maps to a dictionary containing, as the only key, the best action which maps to 1
        INPUT
            policy; dictionary mapping each state to a singleton dictionary, which maps a single action to 1
            value; evaluation of the given policy as a dictionary mapping each state to its value
            disc; discount rate
        RETURNS
            true if no improvements could be made; else false
        '''
        stable = True
        
        # For each state, greedily select best action
        for st in self.game.states:

            # If absorbing state, continue
            if len(self.game.actions[st]) == 0:
                continue

            # Record the old action and trackers for new action
            old_action = list(policy[st].keys())[0]
            best_action = None
            best_action_value = float('-inf')

            # For each action, calculate the value of that action
            for act in self.game.actions[st]:
                action_value = 0
                state_reward_list, prob_state_reward_list = self.game.get_state_and_reward(st, act)
                for state_reward, prob_state_reward in zip(state_reward_list, prob_state_reward_list):

                    # Compute contribution to the value function
                    action_value += prob_state_reward * (state_reward[1] + disc * value[state_reward[0]])

                # Track if new best action is found
                if action_value > best_action_value:
                    best_action = act
                    best_action_value = action_value      
            
            # Update action
            if old_action != best_action:
                stable = False                      
            policy[st] = {best_action: 1}

        return stable
    
    def policy_iteration(self, policy):
        '''
        Executes policy iteration.
        Given a base policy, repeatedly performs evaluation and improvement until no improvement can be made.
        Assumes the base policy is deterministic, where each state is mapped only to a singleton dictionary that maps the selected action to probability 1
        INPUT
            policy; dictionary mapping each state to a singleton dictionary, which maps a single action to 1
        RETURNS 
            best deterministic policy
        '''
        # Copy policy
        policy = policy.copy()

        # Repeatedly evaluate and improve until no improvement is possible
        stable = False
        while not stable:
            value = self.policy_evaluation(policy)
            stable = self.policy_improvement(policy, value)
        return policy

    def value_iteration(self):
        '''
        Executes value iteration
        RETURNS
            best deterministic policy
        '''

        def estimate_state_value(state_reward_pr: Tuple[List[Tuple[int, float]], List[float]]):
            ([(s_next, r)], [pr]) = state_reward_pr
            return pr * (r + self.gamma * V[s_next])

        # Initialize value function
        V = {**{0: 0}, **{st: random.randint(-100, -1) for st in self.game.states[1:]}}

        # Set delta to high number
        delta = np.inf

        # While change in value function is high, loop
        while delta > self.tol:
            delta = 0

            # Compute new values for each state
            for st in self.game.states[1:]:
                v = V[st]
                V[st] = max(estimate_state_value(self.game.get_state_and_reward(st, a)) for a in self.game.actions[st])
                delta = max(delta, abs(v - V[st]))
                #logger.info("Value Iteration", st=st, v=V[st], delta=delta, V=V)

        # Construct policy
        return {**{0: {}}, **{st: {self.game.actions[st][np.argmax([estimate_state_value(self.game.get_state_and_reward(st, a)) for a in self.game.actions[st]])]: 1}
                for st in self.game.states[1:]}}
            


gw = GridWorld()
dp = DynamicProgram(gw)

# Equiprobable random policy evaluation
erp = gw.make_equiprobable_random_policy()
value = dp.policy_evaluation(erp)
print(value)

# Policy iteration

# Construct arbitrary terminable policy: go left until you leftmost column, then go up
pol = {st: {"L": 1} for st in gw.states}
for st in [4, 8, 12]:
    pol[st] = {"U": 1}
pol[0] = {}
tol = 0
dp.tol = tol
best_pol = dp.policy_iteration(pol)
logger.info("Policy Iteration", tol=tol, best_pol=best_pol, value=dp.policy_evaluation(best_pol))

threshold, gamma = 0.0001, 1.
dp.tol = threshold
dp.gamma = gamma
best_pol = dp.value_iteration()
value = dp.policy_evaluation(best_pol)
logger.info("Value Iteration", threshold=threshold, gamma=gamma, best_pol=best_pol, value=value)

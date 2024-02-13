from abc import ABC, abstractmethod
import random
import numpy as np
from typing import List, Tuple

class Game(ABC):
    '''
    Abstract interface for general games
    '''
    states = []     # list of states
    actions = {}    # dictionary mapping each state to a list of possible actions. empty list implies terminal state.

    @abstractmethod
    def get_state_and_reward(self, oldstate, action):
        '''
        Given the current state and an action, outputs two parallel lists describing potential outcomes
        INPUT
            oldstate; the state before the action was taken
            action; the taken action
        RETURNS
            list of 2-tuples, where the 0th element is the potential next state and 1th element is the potential reward
            list of probabilities of achieving the associated state and reward
        '''
        pass

    @abstractmethod
    def make_states(self):
        '''
        Makes the list of states for this game
        '''
        pass

    @abstractmethod
    def make_actions(self):
        '''
        Makes the dictionary mapping states to list of possible actions
        Assumes self.states has been constructed prior to calling make_actions 
        Maps a state to empty list if it is a terminal state
        '''
        pass

    def __init__(self):
        '''
        Initializes this game by making the states and the actions
        '''
        self.make_states()
        self.make_actions()

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
    
    def is_done(self, state):
        '''
        Checks if given state is terminal
        INPUT
            state; a state
        RETURNS
            True if state is terminal, else False
        '''
        return (len(self.actions[state]) == 0)
    
    def sample_policy_action(self, state, policy=None):
        '''
        Returns a sample of an action to perform at this state
        INPUT
            state; the current state
            policy; dictionary mapping each state to a dictionary that maps each action to the probability of selecting that action
                if None, assumes equiprobable random policy
        RETURNS
            an action determined by the policy; if terminal state, returns None
        '''
        # If no policy is provided, sample at random
        if policy is None:
            return np.random.choice(a=self.actions[state])

        # Get list of valid actions
        actions = list(policy[state].keys())

        # If terminal state, return None
        if len(actions) == 0:
            return None

        # Sample policy
        probs = [policy[state][act]
                    for act in actions]
        return np.random.choice(a=actions, p=probs)


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

    def action_evaluation(self, state, action, value):
        '''
        Evaluates a given action at a current state
        INPUT
            state; current state
            action; intended action
            value; dictionary mapping each state to its value
        RETURNS
            value of the action at the currrent state under the given value mapping
        '''
        # Initialize
        action_value = 0

        # Get action outcomes
        state_reward_list, prob_state_reward_list = self.game.get_state_and_reward(state, action)
        
        # Compute value of each outcome
        for state_reward, prob_state_reward in zip(state_reward_list, prob_state_reward_list):
            action_value += prob_state_reward * (state_reward[1] + self.gamma * value[state_reward[0]])

        return action_value

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
        
        # While change in value function is high, loop
        delta = np.inf
        while delta > self.tol:
            delta = 0

            # Store old values for computing change in values this iteration
            old_value = value.copy()

            # Compute new values for each state
            for st in self.game.states:
                if self.game.is_done(st):
                    continue
                value[st] = 0
                for act in policy[st].keys():
                    value[st] += policy[st][act] * self.action_evaluation(st, act, old_value)
                
                # Compute change in this state's value function
                delta = max(delta, abs(old_value[st] - value[st]))
        return value   

    def policy_improvement(self, value, policy=None):
        '''
        Improves the given policy
        Assumes the given policy is deterministic: each state maps to a dictionary containing, as the only key, the best action which maps to 1
        INPUT
            value; evaluation of the given policy as a dictionary mapping each state to its value
            policy; dictionary mapping each state to a singleton dictionary, which maps a single action to 1
                if None, returns false stability flag
        RETURNS
            new policy; a greedily improved policy
            stable; true if no improvements could be made; else false
        '''
        # Greedily find new policy
        new_policy = {st: ({self.game.actions[st][np.argmax([self.action_evaluation(st, act, value) for act in self.game.actions[st]])]: 1}
                        if not self.game.is_done(st) else {})
                        for st in self.game.states}

        # If no base policy was provided, return unstable
        if policy is None:
            return new_policy, False

        # If policy states do not align, return unstable
        if policy.keys() != new_policy.keys():
            return new_policy, False

        # Check for policy stability under each state
        for st in self.game.states:

            # If policy actions to not align, return unstable
            if policy[st].keys() != new_policy[st].keys():
                return new_policy, False
            
            # If policy action probablity do not align, return unstable
            for act in policy[st]:
                if policy[st][act] != new_policy[st][act]:
                    return new_policy, False
                
        # Return stable
        return new_policy, True
    
    def policy_iteration(self, policy=None):
        '''
        Executes policy iteration.
        Given a base policy, repeatedly performs evaluation and improvement until no improvement can be made.
        INPUT
            policy; dictionary mapping each state to a singleton dictionary, which maps a single action to 1
                if None, uses the equiprobable random policy as the base policy
        RETURNS 
            best deterministic policy
        '''
        # Default policy to equiprobable random policy
        if policy is None:
            policy = self.game.make_equiprobable_random_policy()

        # Repeatedly evaluate and improve until no improvement is possible
        stable = False
        while not stable:
            value = self.policy_evaluation(policy)
            policy, stable = self.policy_improvement(value, policy)
        return policy

    def value_iteration(self):
        '''
        Executes value iteration
        RETURNS
            best deterministic policy
        '''
        # Initialize value function
        V = {st: 0 for st in self.game.states}

        # While change in value function is high, loop
        delta = np.inf
        while delta > self.tol:
            delta = 0

            # Compute new values for each state
            for st in self.game.states:
                if self.game.is_done(st):
                    continue
                v = V[st]
                V[st] = max(self.action_evaluation(st, act, V) for act in self.game.actions[st])
                delta = max(delta, abs(v - V[st]))

        # Construct policy
        return {st: ({self.game.actions[st][np.argmax([self.action_evaluation(st, act, V) for act in self.game.actions[st]])]: 1}
                if not self.game.is_done(st) else {})
                for st in self.game.states}
            
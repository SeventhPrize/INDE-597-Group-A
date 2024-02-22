from abc import abstractclassmethod
from typing import List
import numpy as np
from collections import defaultdict
from environments.competition import Agent


class TemporalDifference(Agent):
    '''
    Interface for temporal difference learning agent
    Makes epsilon-greedy policy for soft behavior
    '''
    Q = defaultdict(float)  # dictionary that maps a (state, action) 2-tuple to estimated value
    eps = 1                 # epsilong parameter for epsilon-greedy policy
    gamma = 1               # discount factor
    alpha = 1               # learning rate

    def __init__(self, eps=1, gamma=1, alpha=1):
        '''
        Initializes this agent
        INPUT
            eps; epsilon parameter for epsilon-greedy policy
            gamma; discount factor
            alpha; learning rate
        '''
        super().__init__()
        self.Q = defaultdict(float)
        self.eps = eps
        self.gamma = gamma
        self.alpha = alpha

    def play(self, state):
        '''
        Plays according to epsilon-greedy from given state
        '''
        # Explore
        if np.random.random() < self.eps:
            return np.random.choice(self.env.get_actions())
        
        # Exploit
        else:
            act_ind = np.argmax([self.Q[(state, act)] for act in self.env.get_actions()])
            return self.env.get_actions()[act_ind]
        
    @abstractclassmethod
    def see_history(self, history:List):
        '''
        Trains from given history
        INPUT
            history; list of 3-tuples of structure
                0: state
                1: action
                2: reward
        '''
        pass

class Sarsa(TemporalDifference):
    '''
    Trains agent using SARSA
    '''
    def see_history(self, history):
        '''
        Trains from given history using SARSA
        INPUT
            history; list of 3-tuples of structure
                0: state
                1: action
                2: reward
        '''
        # If history does not have two state/action pairs, do nothing
        if len(history) < 2:
            return
        
        # Get the transition
        old_pair = (history[-2][0], history[-2][1])
        new_pair = (history[-1][0], history[-1][1])
        reward = history[-2][2]
        
        # Update Q
        self.Q[old_pair] += self.alpha * (reward + self.gamma * self.Q[new_pair] - self.Q[old_pair]) 
        
class QLearning(TemporalDifference):
    '''
    Trains agent using Q-Learning
    '''
    def see_history(self, history):
        '''
        Trains from given history using Q-Learning
        INPUT
            history; list of 3-tuples of structure
                0: state
                1: action
                2: reward
        '''
        # If history does not have two state/action pairs, do nothing
        if len(history) < 2:
            return
        
        # Get the transition
        old_pair = (history[-2][0], history[-2][1])
        new_state = history[-1][0]
        reward = history[-2][2]
        
        # Update Q
        new_state_value = np.max([self.Q[(new_state, act)] for act in self.env.get_actions()])
        self.Q[old_pair] += self.alpha * (reward + self.gamma * new_state_value - self.Q[old_pair]) 

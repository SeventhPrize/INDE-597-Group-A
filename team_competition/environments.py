from abc import ABC, abstractmethod
from typing import List, Tuple
import random

class GenericEnvironment(ABC):
    '''
    Interface for a team-vs-team environment
    '''
    agents = tuple()        # 2-tuple of agents
    current_state = None    # current state of this environment

    def __init__(self, agents:Tuple):
        '''
        Initializes this environment by setting the agents
        '''
        if len(agents) != 2:
            raise Exception("Expected 2 agents.")
        
        self.agents = tuple(agents)
        for agent in self.agents:
            agent.associate_environment(self)

    @abstractmethod
    def get_actions(self):
        '''
        RETURNS iterable of all actions
        '''
        return tuple()
    
    @abstractmethod
    def step(self, action, agent_ind:int):
        '''
        Steps in the current game
        Mutates self.current_state to be the next state
        INPUT
            action; action taken at this step
            agent_ind; index of the agent who took this action
        RETURNS 4 arguments
            0: next state after the step
            1: reward for the action
            2: boolean flag whether the environment has terminated
            3: the index of the agent whose turn it is
        '''
        return None, 0, False

    @abstractmethod
    def reset(self):
        '''
        Resets this envinroment
        RETURNS
            the next starting state
        '''
        return None

    @abstractmethod
    def is_terminal_state(self, state):
        '''
        RETURNS if the given state is a terminal state
        '''
        return False

    @abstractmethod
    def reinterpret_state_for_agent(self, state, agent_ind:int):
        '''
        Reinterprets the given state for the given indexed agent
        INPUT
            state; current state
            agent_ind; the index of the agent to reinterpret the state for; either 0 or 1
        RETURNS
            the state reinterpreted for the agent
        '''
        return None
    
    @abstractmethod
    def render(self, state):
        '''
        RETURNS figure of the current state
        '''
        return None

    def play_game(self, first_agent_ind:int=random.randint(0, 1)):
        '''
        Plays the game and outputs the episode path
        RETURNS
            episode path; list of 4-tuples, each of which has components:
                0: state
                1: action
                2: reward
                3: index of agent
                The last element is the final state, given as (final_state, None, None, None)
            rewards; 2-length list of each agent's final rewards
        '''
        episode_path = []
        rewards = [0, 0]
        agent_ind = first_agent_ind
        state = self.reset()
        while True:
            reinterpret_state = self.reinterpret_state_for_agent(state, agent_ind)
            action = self.agents[agent_ind].compute_action(reinterpret_state, agent_ind)
            next_state, reward, done, next_agent_ind = self.step()
            episode_path.append((state, action, reward, agent_ind))
            rewards[agent_ind] += reward
            state = next_state
            agent_ind = next_agent_ind
            if done:
                episode_path.append((state, None, None, None))
                return episode_path, rewards

class GenericTabularEnvironment(GenericEnvironment):
    '''
    Interface for a team-vs-team environment
    that has an enumerable state space
    '''
    @abstractmethod
    def get_states(self):
        '''
        RETURNS iterable of all states
        '''
        return tuple()
    
class Agent(ABC):
    '''
    Interface for a game agent
    '''
    env = None  # the environment associated with this agent

    @abstractmethod
    def compute_action(self, state):
        '''
        INPUT
            state; the current state
        RETURNS
            the action taken by this agent
        '''
        return None

    def associate_environment(self, env:GenericEnvironment):
        '''
        Assigns an environment to this agent
        '''
        self.env = env
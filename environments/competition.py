from abc import ABC, abstractmethod
from typing import Sequence, List, Tuple
from collections.abc import Hashable
import random

class EnvironmentVersus(ABC):
    '''
    Interface for a team-vs-team environment
    '''
    agents = tuple()        # tuple of agents
    current_state = None    # current state

    def __init__(self, agents:Sequence):
        '''
        Initializes this environment by setting the agents
        INPUT
            list/tuple of agents
        '''        
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
        return None, 0, False, None

    @abstractmethod
    def reset(self):
        '''
        Resets this envinroment
        RETURNS
            the next starting state
        '''
        return None

    @abstractmethod
    def reinterpret_state_for_agent(self, state:Hashable, agent_ind:int):
        '''
        Reinterprets the given state for the given indexed agent
        INPUT
            state; current state
            agent_ind; the index of the agent to reinterpret the state for; either 0 or 1
        RETURNS
            the state reinterpreted for the agent
        '''
        return None
            
    def play_game(self, first_agent_ind:int=-1):
        '''
        Plays the game and outputs the episode path
        INPUT
            first_agent_ind; integer index of the agent to make the first move
                if -1, randomly selects an agent
        RETURNS
            episode path; list of 4-tuples, each of which has components:
                0: state
                1: action
                2: reward
                3: index of agent
                The last element is the final state, given as (final_state, None, None, None)
            rewards; 2-length list of each agent's final rewards
        '''
        # Select (default) agent
        if first_agent_ind == -1:
            agent_ind = random.choice(range(len(self.agents)))
        else:
            agent_ind = first_agent_ind

        # Initialize
        episode_path = []
        individual_paths = [[]] * len(self.agents)
        rewards = [0] * len(self.agents)
        state = self.reset()

        # Until done, make steps and record action
        while True:

            # Make state interpretable to the agent whose turn it is
            reinterpret_state = self.reinterpret_state_for_agent(state, agent_ind)
            
            # Get agent's action
            action = self.agents[agent_ind].play(reinterpret_state)
            
            # Get outcome
            next_state, reward, done, next_agent_ind = self.step(action, agent_ind)
            episode_path.append((state, action, reward, agent_ind))
            individual_paths[agent_ind].append((self.reinterpret_state_for_agent(state, agent_ind), action, reward))
            rewards[agent_ind] += reward
            state = next_state
            agent_ind = next_agent_ind

            # Add termination state
            if done:
                episode_path.append((state, None, 0, None))
                for agent_ind in range(len(self.agents)):
                    individual_paths.append((self.reinterpret_state_for_agent(state, agent_ind), action, 0))

            # Show history to agents
            for agent_ind, agent in enumerate(self.agents):
                agent.see_history(individual_paths[agent_ind])

            # Terminate
            if done:
                return episode_path, rewards
            
class Agent(ABC):
    '''
    Interface for a game agent
    '''
    env = None  # the environment associated with this agent

    def __init__(self):
        '''
        Initializes the agent
        '''
        pass

    @abstractmethod
    def play(self, state:Hashable):
        '''
        INPUT
            state; the current state
        RETURNS
            the action taken by this agent
        '''
        return None

    def associate_environment(self, env:EnvironmentVersus):
        '''
        Assigns an environment to this agent
        '''
        self.env = env

    def see_history(self, history:List):
        '''
        Does NOT have to be implemented.

        Implement to learn from given history, e.g. for training.
        
        INPUT
            history; list of 3-tuples of structure
                0: state
                1: action
                2: reward
        '''
        pass
from abc import ABC, abstractmethod
from typing import Sequence, List, Tuple
import random

class EnvironmentVersus(ABC):
    '''
    Interface for a team-vs-team environment
    '''
    agents = tuple()        # 2-tuple of agents
    current_state = None    # current state

    def __init__(self, agents:Sequence):
        '''
        Initializes this environment by setting the agents
        INPUT
            2-length list/tuple/set of agents
        '''
        # Check for two agents
        if len(agents) != 2:
            raise Exception("Expected 2 agents.")
        
        # Set agents
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
        # Initialize
        episode_path = []
        individual_paths = [[][]]
        rewards = [0, 0]
        agent_ind = first_agent_ind
        state = self.reset()

        # Until done, make steps and record action
        while True:

            # Make state interpretable to the agent whose turn it is
            reinterpret_state = self.reinterpret_state_for_agent(state, agent_ind)
            
            # Get agent's action
            action = self.agents[agent_ind].compute_action(reinterpret_state, agent_ind)
            
            # Get outcome
            next_state, reward, done, next_agent_ind = self.step()
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

    @abstractmethod
    def play(self, state):
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
        Processes given history
        INPUT
            history; list of 3-tuples of structure
                0: state
                1: action
                2: reward
        '''
        pass
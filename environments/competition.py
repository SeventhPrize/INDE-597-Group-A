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
            
    def play_game(self, opening:Sequence=None):
        '''
        Plays the game and outputs the episode path
        INPUT
            opening; a 2-tuple with elements
                0: initial state
                1: first agent to move

                if default None, then sets initial state using self.reset() and initial agent randomly  
            
        RETURNS
            episode path; list of 4-tuples, each of which has components:
                0: state
                1: action
                2: reward
                3: index of agent
                The last element is the final state, given as (final_state, None, None, None)
            rewards; 2-length list of each agent's final rewards
        '''
        # Set opening default by reset
        if opening is None:
            state = self.reset()
            agent_ind = random.choice(range(len(self.agents)))

        # Set given opening
        else:
            (state, agent_ind) = opening
            self.current_state = state

        # Initialize
        episode_path = []
        individual_paths = [[] for _ in range(len(self.agents))]
        rewards = [0] * len(self.agents)

        # Until done, make steps and record action
        while True:

            # Make state interpretable to the agent whose turn it is
            reinterpret_state = self.reinterpret_state_for_agent(state, agent_ind)
            
            # Get agent's action
            action = self.agents[agent_ind].play(reinterpret_state)
            
            # Get outcome
            next_state, reward, done, next_agent_ind = self.step(action, agent_ind)
            episode_path.append((state, action, reward, agent_ind))
            individual_paths[agent_ind].append((reinterpret_state, action, reward))
            rewards[agent_ind] += reward
            state = next_state
            agent_ind = next_agent_ind

            # Add termination state
            if done:
                episode_path.append((state, None, 0, None))

                # Modify each agent's last action based on the end-game rewards
                for agent_ind_loop in range(len(self.agents)):
                    if len(individual_paths[agent_ind_loop]) == 0:
                        continue
                    last_sar = list(individual_paths[agent_ind_loop][-1])
                    reward_change = self.compute_game_end_reward(episode_path, agent_ind_loop)
                    last_sar[2] += reward_change
                    individual_paths[agent_ind_loop][-1] = tuple(last_sar)
                    rewards[agent_ind_loop] += reward_change

                # Give each agent the termination state
                for agent_ind_loop in range(len(self.agents)):
                    individual_paths[agent_ind_loop].append((self.reinterpret_state_for_agent(state, agent_ind_loop), action, 0))

            # Show history to agents
            for agent_ind_loop, agent in enumerate(self.agents):
                agent.see_history(individual_paths[agent_ind_loop])

            # Terminate
            if done:
                return episode_path, rewards
            
    def compute_game_end_reward(self, history:List, agent_ind:int):
        '''
        Computes the reward accrued by the given indexed agent at the end of the game
        
        This implementation assumes a zero-sum game: gives the indexed reward equal to the negative of the reward gained by the last action played.
        This implementation does not indicate changes to the reward of the agent who played the last action, since that agent already got reward from their action.
        Consider alternate implementations for different games.
        
        INPUT
            history; episode pathway as a list of 4-tuples, each of which has components:
                0: state
                1: action
                2: reward
                3: index of agent
                The last element is the final state, given as (final_state, None, None, None)
            agent_ind; integer index of the agent to compute the game end rewards for
        RETURNS
            amount by which to modify the reward of the last action made by the indexed agent
        '''
        # Check that the last element in the pathway is a final_state,
        # indicated by None as the index of the agent
        if history[-1][3] is not None:
            raise Exception("The last episode pathway element had a non-None agent_ind, indicating the episode is not yet done.")

        # Get agent and reward of the last action
        last_agent_ind = history[-2][3]
        last_reward = history[-2][2]

        # If the last agent and the given agent are the same, return 0
        if last_agent_ind == agent_ind:
            return 0
        
        # Return the negative of the reward gained by the last agent
        return -last_reward
            
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
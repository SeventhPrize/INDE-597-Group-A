from abc import abstractmethod
from environments.competition import *

class EnvironmentVersusSelf(EnvironmentVersus):
    '''
    Interface for training an agent for team vs team competition
    by playing the agent against itself.
    '''
    def __init__(self, agent:Agent, n_player:int=2):
        '''
        Initializes this environment by giving only the agent to the superclass
        INPUT
            agent; the agent to train
            n_player; number of players in this game
        '''
        super().__init__(tuple([agent] * n_player))

class EnvironmentSolitaire(EnvironmentVersus):
    '''
    Interface for a solitaire single-player game
    '''
    def __init__(self, agent:Agent):
        '''
        Initializes this environment with only one agent.
        '''
        super().__init__(tuple([agent]))

    @abstractmethod
    def step(self, action):
        '''
        Overloads the step function to only consider the single agent for easier syntax
        Mutates self.current_state to be the next state
        INPUT
            action; action taken at this step
        RETURNS 3 arguments
            0: next state after the step
            1: reward for the action
            2: boolean flag whether the environment has terminated
        '''
        return None, 0, False
    
    def play_game(self):
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
        individual_paths = [[]]
        rewards = [0]
        state = self.reset()

        # Until done, make steps and record action
        while True:

            # Get agent's action
            action = self.agents[0].play(state)
            
            # Get outcome
            next_state, reward, done = self.step(action)
            episode_path.append((state, action, reward, 0))
            individual_paths[0].append((state, action, reward))
            rewards[0] += reward
            state = next_state

            # Add termination state
            if done:
                episode_path.append((state, None, 0, None))
                individual_paths[0].append((state, action, 0))

            # Show history to agents
            self.agents[0].see_history(individual_paths[0])

            # Terminate
            if done:
                return episode_path, rewards
    
    def reinterpret_state_for_agent(self, state:Hashable):
        '''
        Returns the given state
        INPUT
            state; current state
        RETURNS
            the state reinterpreted for the agent
        '''
        return state


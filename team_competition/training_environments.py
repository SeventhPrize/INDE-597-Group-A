from abc import abstractmethod
from team_competition.environments import *

class EnvironmentVersusSelf(EnvironmentVersus):
    '''
    Interface for training an agent for team vs team competition
    by playing the agent against itself.
    '''
    def __init__(self, agent:Agent):
        '''
        Initializes this environment by giving only the agent to the superclass
        '''
        super().__init__((agent, agent))


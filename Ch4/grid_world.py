from dynamic_program import Game, DynamicProgram
import structlog
from typing import List, Tuple

logger = structlog.get_logger()


class GridWorld(Game):
    '''
    Implements the GridWorld game
    '''
    def make_states(self):
        '''
        Makes the 16 states
        '''
        self.states = list(range(15))

    def make_actions(self):
        '''
        Makes the UDLR actions 
        '''
        self.actions = {st : ["U", "D", "L", "R"] for st in self.states}
        self.actions[0] = []

    def get_state_and_reward(self, oldstate, action):
        '''
        Given the current state and an action, returns two parallel lists describing the outcome
        For GridWorld, the lists are all length 1 because the actions all have deterministic outcome
        INPUT
            oldstate; the state before the action was taken
            action; the taken action
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

gw = GridWorld()
dp = DynamicProgram(gw)

# Equiprobable random policy evaluation
erp = gw.make_equiprobable_random_policy()
value = dp.policy_evaluation(erp)
logger.info("ERP Evaluation", value=value)

# Policy iteration
pol = dp.policy_iteration(None)
logger.info("Policy Iteration", policy=pol, value=dp.policy_evaluation(pol))

pol = dp.value_iteration()
logger.info("Value Iteration", policy=pol, value=dp.policy_evaluation(pol))
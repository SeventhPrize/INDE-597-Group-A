import gymnasium as gym
from gymnasium.spaces import Tuple, Discrete
import matplotlib.pyplot as plt
import numpy as np
import itertools
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from dynamic_program import Game, DynamicProgram

class ToyTextGym(Game):
    '''
    Wraps an OpenAI Gym Toy Text game as a Game object
    '''
    env = None  # The OpenAI gym environment associated with this game

    def __init__(self, toy_text_gym_env):
        '''
        Initializes this game
        INPUT
            toy_text_gym_env; an OpenAI toy text gym environment
        '''
        # Set environment
        self.env = toy_text_gym_env

        # Make states and actions
        super().__init__()

    def unpack_space(self, space):
        '''
        Unpacks the given Gymnasium space for purposes of enumerating states/actions
        Assumes the space is a Discrete object or a Tuple of Discrete objects
        INPUT
            space; Gymnasium Discrete or Tuple of Discretes
        RETURNS
            enumeration of all items in the space
        '''
        # If Discrete, return list of integers within Discrete
        if isinstance(space, Discrete):
            return list(range(space.n))
        
        # If Tuple, return cartesian product of Discrete bounds
        elif isinstance(space, Tuple):

            # Check that elements are Discrete
            for elm in space:
                if not isinstance(elm, Discrete):
                    raise TypeError("Gymnasium Tuple contains types other than Discrete")
            
            # Return cartesian product of Discrete bounds
            return list(itertools.product(*[range(elm.n) for elm in space]))
        
        # Raise error
        raise TypeError("Gymnasium space is not a Tuple nor a Discrete")

    def make_states(self):
        '''
        Makes the states
        '''
        self.states = self.unpack_space(self.env.observation_space)

    def make_actions(self):
        '''
        Makes the actions, a dictionary mapping each state to each possible action
        '''
        # Make actions
        actions = self.unpack_space(self.env.action_space)
        self.actions = {st: actions for st in self.states}
        
        # For each action that leads to a termination, record the terminal state
        for st in self.states:
            for act in self.actions[st]:
                for step_outcome in self.env.unwrapped.P[st][act]:
                    if step_outcome[3]:
                        self.actions[step_outcome[1]] = []
        
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
        # Initialize lists describing potential outcomes
        state_reward_list = []
        prob_list = []

        # Record each potential outcome
        for outcome in self.env.unwrapped.P[oldstate][action]:
            state_reward_list.append((outcome[1], outcome[2]))
            prob_list.append(outcome[0])
        
        # Return
        return state_reward_list, prob_list 

    def reset_env(self):
        '''
        Resets the gym environment
        RETURNS
            the starting state that was reset to for the next iteration
        '''
        reset_state, _ = self.env.reset()
        return reset_state
    
    def policy_animation(self, policy=None):
        '''
        Displays an animation for a single episode under the given policy
        Animation code from Dr. Davila
        INPUT
            policy; dictionary mapping each state to a dictionary that maps each action to the probability of selecting that action
                if None, assumes equiprobable random policy
        RETURNS
            animation frames as list of renderings
        '''
        # Reset and initialize
        state = self.reset_env()
        frames = [self.env.render()]

        # Sample actions from the policy until a terminal state is reached
        while not self.is_done(state):
            action = self.sample_policy_action(state, policy)
            state, _, _, _, _ = self.env.step(action)
            frames.append(self.env.render())

        # Make figure for animation
        fig, ax = plt.subplots()

        # Function to update the figure with the new frame
        def update(frame):
            ax.clear()  # Clear previous frame
            ax.imshow(frame)  # Display the current frame
            return ax,

        # Creating the animation
        ani = FuncAnimation(fig, update,
                            frames=frames,
                            blit=False,
                            interval=50,
                            repeat_delay=500)
        plt.show()
        # HTML(ani.to_html5_video())                

def toy_text_experiment(mdl_name, n_animations=1):
    env = gym.make(mdl_name, render_mode="rgb_array")
    game = ToyTextGym(env)
    dp = DynamicProgram(game)    
    dp.tol = 0.01
    pol = dp.value_iteration()
    for _ in range(n_animations):
        game.policy_animation(pol)

for mdl in ["FrozenLake-v1",
            # "Blackjack-v1",
            "CliffWalking-v0",
            "Taxi-v3"]:
    toy_text_experiment(mdl, 3)
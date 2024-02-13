import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from dynamic_program import Game, DynamicProgram

class ToyTextGym(Game):
    '''
    Wraps an OpenAI Gym Toy Text game as a Game object
    '''
    env = None              # The OpenAI gym environment associated with this game

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
        
    def make_states(self):
        '''
        Makes the states
        '''
        self.states = list(range(self.env.observation_space.n))

    def make_actions(self):
        '''
        Makes the actions, a dictionary mapping each state to each possible action
        '''
        # Make actions
        self.actions = {st: list(range(self.env.action_space.n))
                        for st in self.states}
        
        # Record terminal states: For each state, check if any actions lead to new states.
        for st in self.states:
            is_done = True
            for act in self.actions[st]:
                for step_outcome in self.env.unwrapped.P[st][act]:
                    new_state = step_outcome[1]
                    if new_state != st:
                        is_done = False

            # If the state cannot be acted upon to move to a different state, mark as terminal
            if is_done:
                self.actions[st] = []
        
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
                            interval=100,
                            repeat_delay=500)
        plt.show()
        # HTML(ani.to_html5_video())                

frozen_lake = ToyTextGym(gym.make("FrozenLake-v1", render_mode="rgb_array"))
dp = DynamicProgram(frozen_lake)
# pol = dp.value_iteration()
pol = dp.policy_iteration()
frozen_lake.policy_animation(dp.policy_iteration(pol))
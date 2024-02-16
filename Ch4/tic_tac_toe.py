import itertools
import pickle
import time
from copy import copy
from typing import Dict, List, Tuple

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

from structlog import get_logger

logger = get_logger(__name__)

class TicTacToeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}


    def __init__(self, render_mode="rgb_array"):
        self.size = 3 # The size of the grid
        self.window_size = 512  # The size of the PyGame window

        self.board = [0]*9  # The board is represented as a 1D list of 9 elements
        self.render_turn = 1  # The turn of the agent that is currently being rendered
        self.prev_board = None  # The board that was rendered in the previous frame

        # Observation space is a 3x3 grid, with 3 possible values for each cell
        # (0 for empty, 1 for player 1, 2 for player 2)
        self.observation_space = spaces.MultiDiscrete([3]*9)

        # Action space is a 3x3 grid, with 9 possible actions (To play in any given cell)
        self.action_space = spaces.Discrete(9)

        self.render_mode = render_mode

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        self.P = self._create_state_action_to_update()

    def _create_state_action_to_update(self) -> Dict[Tuple[int], Dict[int, List[Tuple[float, Tuple[int], int, bool]]]]:
        """
        Create a dictionary that maps each state to a dictionary that maps each action to a list of tuples
        Each tuple contains:
            the probability of transitioning to a new state (float),
            the new state (tuple[int]),
            the reward (int 0, 1, -1),
            and whether the new state is terminal (bool)
        RETURNS
            dictionary that maps each state to a dictionary that maps each action to a list of tuples
        """

        try:
            # If the dictionary has been pickled, load it to save time
            logger.info("Loading state_action_to_update_function.pkl")
            with open("state_action_to_update_function.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            logger.info("It can not be loaded. Computing state_action_to_update_function.pkl")

        p = {}
        for st in list(itertools.product(*[range(3) for elm in range(9)])):
            p[st] = {}
            for act in range(9):
                p[st][act] = []  # prob, state, reward, terminated
                for oponent_action in range(9):
                    try:
                        board, reward, terminated, _, _ = self._step(st, act, oponent_action)
                        p[st][act].append((1, board, reward, terminated))
                    except AssertionError:
                        logger.info(f"Invalid action {act} in state {st}")
                if p[st][act]:
                    p[st][act] = [(t[0] / len(p[st][act]), t[1], t[2], t[3]) for t in p[st][act]]

        with open("state_action_to_update_function.pkl", "wb") as f:
            logger.info("Saving state_action_to_update_function.pkl")
            pickle.dump(p, f)

        return p

    def _get_observation(self):
        return tuple(self.board)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.board = [0]*9
        self.prev_board = None
        observation = self._get_observation()

        if self.render_mode == "human":
            self._render_frame()

        return observation, {}

    def _step(self, board, action, opponent_action):
        assert board[action] == 0
        board = list(board)
        board[action] = 1
        # An episode is done iff the agent has reached the target
        terminated = self._has_tree_in_a_row(board)
        if terminated:
            reward = 1
        else:
            # Opponents play
            assert board[opponent_action] == 0
            board[opponent_action] = 2
            terminated = self._has_tree_in_a_row(board)
            reward = 0 if not terminated else -1

        return tuple(board), reward, terminated, False, {}

    def _has_tree_in_a_row(self, board):
        logger.info(f"Checking if there is a tree in a row")
        # Check rows
        for i in range(3):
            if board[3*i] == board[3*i+1] == board[3*i+2] != 0:
                return True
        # Check columns
        for j in range(3):
            if board[j] == board[3+j] == board[6+j] != 0:
                return True
        # Check diagonals
        if board[0] == board[4] == board[8] != 0:
            return True
        if board[2] == board[4] == board[6] != 0:
            return True
        return False

    def _oponent_play(self, board):
        # random oponent
        empty = [i for i in range(len(board)) if board[i] == 0]
        i = np.random.choice(empty)
        return i

    def render(self):
        if self.render_mode == "rgb_array":
            if self.prev_board is None:
                self.prev_board = copy(self.board)
                return [self._render_frame()]
            else:
                # Split the player's and the opponent's moves into separate frames for visualization purposes
                diff_board = [self.board[i] if self.board[i] != self.prev_board[i] else 0 for i in
                              range(len(self.board))]
                logger.info("diff_board", diff_board=diff_board, prev_board=self.prev_board, board=self.board)
                idx_player_1 = diff_board.index(1)
                try:
                    idx_player_2 = diff_board.index(2)
                except ValueError:
                    idx_player_2 = None

                self.board = self.prev_board
                self.board[idx_player_1] = 1
                f1 = self._render_frame()
                if idx_player_2 is not None:
                    self.board[idx_player_2] = 2
                    f2 = self._render_frame()

                self.prev_board = copy(self.board)
                return [f1, f2] if idx_player_2 is not None else [f1]

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Draw cross and circles
        for i in range(3):
            for j in range(3):
                if self.board[i*3 + j] == 1:
                    pygame.draw.circle(
                        canvas,
                        (0, 0, 255),
                        (i * pix_square_size + pix_square_size // 2, j * pix_square_size + pix_square_size // 2),
                        pix_square_size // 3,
                        5,
                    )
                elif self.board[i*3 + j] == 2:
                    pygame.draw.line(
                        canvas,
                        (255, 0, 0),
                        (i * pix_square_size + pix_square_size // 4, j * pix_square_size + pix_square_size // 4),
                        (i * pix_square_size + 3 * pix_square_size // 4, j * pix_square_size + 3 * pix_square_size // 4),
                        5,
                    )
                    pygame.draw.line(
                        canvas,
                        (255, 0, 0),
                        (i * pix_square_size + 3 * pix_square_size // 4, j * pix_square_size + pix_square_size // 4),
                        (i * pix_square_size + pix_square_size // 4, j * pix_square_size + 3 * pix_square_size // 4),
                        5,
                    )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def step(self, action):
        reward = -1
        terminated = False
        if self.board[action] == 0:
            self.board[action] = 1
            if self.render_mode == "human":
                self._render_frame()

            # An episode is done iff the agent has reached the target
            terminated = self._has_tree_in_a_row(board=self.board)
            if terminated:
                reward = 1
            else:
                # Opponents play
                i = self._oponent_play(self.board)
                assert self.board[i] == 0
                self.board[i] = 2
                terminated = self._has_tree_in_a_row(board=self.board)
                reward = 0 if not terminated else -1

        observation = self._get_observation()

        if self.render_mode == "human":
            self._render_frame()
            time.sleep(2)

        return observation, reward, terminated, False, {}

register(
    id="gym_examples/Tic-Tac-Toe-v0",
    entry_point="gym_examples.envs:TicTacToeEnv",
    max_episode_steps=300,
)
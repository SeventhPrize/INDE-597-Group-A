# add path to sys.path
import os
import sys 

path_to_this_file = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(path_to_this_file, '..', "Play_GO"))

import json

import structlog
import copy
from board import Board, InvalidMoveBoardException, Play, Skip
import random
import re
from openai import OpenAI

from MCTS import MCTS
from go.GoGame import GoGame
from go.pytorch.NNet import NNetWrapper as NNet
from utils import *
import numpy as np



logger = structlog.get_logger(__name__)


class PlayerException(Exception):
    def __init__(self, message):
        self.message = message


class Player:
    def __init__(self, name, color, size=5):
        self.name = name
        self.color = color
        self.size = size

    def get_next_play(self, board: Board) -> Play:
        raise NotImplementedError()

    def update_reward(self, board: Board, last_play: Play):
        pass


class HumanPlayer(Player):
    def __init__(self, color):
        super().__init__("HumanPlayer", color)

    def get_next_play(self, board: Board) -> Play:
        board_copy = Board(board=copy.deepcopy(board.board))

        while True:
            try:
                row = input("Row: ")
                if "pass" in row.lower():
                    logger.info(f"{self.name} passed")
                    return Skip()
                row = int(row)
                col = int(input("Col: "))
                logger.info(f"{self.name} played", row=row, col=col)
                board_copy.perform_move(Play(row, col), self.color)
                return Play(row, col)
            except ValueError:
                logger.error("Invalid input")
            except InvalidMoveBoardException:
                logger.error("Invalid move", error=e)
            except KeyboardInterrupt:
                exit(0)


class RandomPlayer(Player):
    def __init__(self, color):
        super().__init__("Random Player", color)

    def get_next_play(self, board: Board) -> Play:
        available_moves = [Play(row=i, col=j) for i, j in board.get_empty_cells()] + [Skip()]
        board_copy = Board(board=copy.deepcopy(board.board))

        while available_moves:
            play_idx = random.choice(range(len(available_moves)))
            play = available_moves.pop(play_idx)
            try:
                board_copy.perform_move(play, self.color)
                return play
            except InvalidMoveBoardException:
                continue

        raise PlayerException("No available moves")


from openai import OpenAI

class ChatGPTPlayer(Player):
    text = """
This is a board a 6x6 GO board, and the stones are represented by 'B' for black and 'W' for white. The '#' symbols represents empty intersections. 

%s


Assuming the rows are labeled A-F, and the columns are numbered 1-6, Which do you think will be the best Black's next move?
You must give an answer in the format {row name}{column number}.
    """

    def __init__(self, color):
        super().__init__("ChatGPT Player", color)
        #self.client = OpenAI(api_key='dummy')
        self.client = OpenAI(api_key="dummy")
    def get_next_play(self, board: Board) -> Play:
        logger.info("Asking ChatGPT what to play")
        completion = self.client.chat.completions.create(
            model="gpt-4",
            #model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": self.text % board},
            ]
        )

        available_moves = [Play(row=i, col=j) for i, j in board.get_empty_cells()]
        board_copy = Board(board=copy.deepcopy(board.board))

        answer = completion.choices[0].message
        logger.info("ChatGPT answer", content=answer)
        play = self.get_move_from_answer(answer.content)
        if play is not None:
            available_moves.append(play)
        while available_moves:
            play = available_moves.pop()
            try:
                board_copy.perform_move(play, self.color)
                return play
            except InvalidMoveBoardException:
                continue
        return Skip()

        raise PlayerException("No available moves")

    def get_move_from_answer(self, answer: str):
        # search for the following "Black's next move: E5"
        regex = "([A|B|C|D|E|F])([1|2|3|4|5|6])"
        match = re.search(regex, answer)
        if match:
            row, col = ['A', 'B', 'C', 'D', 'E', 'F'].index(match.groups()[0]), int(match.groups()[1])
            logger.info("ChatGPT suggest to play", row=row, col=col)
            return Play(row, col)
        logger.info("ChatGPT did not choose a play")


class RLPlayer(Player):

    def __init__(self, color):
        super().__init__("RLPlayer", color)
        self.state_action_function = {} #state, action, estimate of future rewards # 3**36
        self.number_of_matches = 1
        self.epsilon = .3

    def get_next_play(self, board: Board) -> Play:
        try:
            actions = self.state_action_function[board.get_hash()]
            epsilon = self.epsilon / sum([n for n, r in self.state_action_function[board.get_hash()].values()])
            if random.random() < epsilon:
                play = max(actions.keys(), key=lambda k: actions[k][1])
                return play
        except KeyError:
            self.state_action_function[board.get_hash()] = {}

        available_moves = [Play(row=i, col=j) for i, j in board.get_empty_cells()] + [Skip()]
        board_copy = Board(board=copy.deepcopy(board.board))
        while available_moves:
            #pop at random and remove
            play = random.choice(range(len(available_moves)))
            play = available_moves.pop(play)
            try:
                board_copy.perform_move(play, self.color)
            except InvalidMoveBoardException:
                continue
            try:
                self.state_action_function[board.get_hash()][play]
            except KeyError:
                self.state_action_function[board.get_hash()][play] = (0, 0)
            return play
        raise PlayerException("No available moves")

    def update_reward(self, board: Board, last_play: Play):
        n, rew_for_board_and_last_play = self.state_action_function[board.get_hash()][last_play]
        n += 1
        if board.get_winner() is None:
            self.state_action_function[board.get_hash()][last_play] += (n, 1/n * (- rew_for_board_and_last_play))
        else:
            # Game is finished
            self.number_of_matchs += 1
            if board.get_winner() == self.color:
                self.state_action_function[board.get_hash()][last_play] += (n, 1/n * (1 - rew_for_board_and_last_play))
            else:
                self.state_action_function[board.get_hash()][last_play] += (n, -1/n * (1 + rew_for_board_and_last_play))
            self.save_state_action_function()

    def save_state_action_function(self):
        with open(f"state_action_function_number_of_matches={self.number_of_matches}.json", "w") as f:
            json.dump(self.state_action_function, f)
            
            
            
class MCTSPlayer(Player):
    def __init__(self, color, size=5):

        super().__init__("MCTSPlayer", color, size=size)

        path_to_this_file = os.path.abspath(os.path.dirname(__file__))
        player_path = os.path.join(path_to_this_file, '..', 'Play_GO', f'trained_{self.size}x{self.size}')


        go = GoGame(size)
        # nn trained player
        nn_bot = NNet(go)
        logger.info("Loading checkpoint", fpath=player_path)

        nn_bot.load_checkpoint(player_path, 'best.pth.tar')
        args = dotdict({'numMCTSSims': 10, 'cpuct':1.0})
        self.mcts = MCTS(go, nn_bot, args)
        
        print(go.getInitBoard())


    def translate_board(self, board):
        board = np.array([[1 if board[i][j] == 'B' else (-1 if board[i][j] == 'W' else 0) for j in range(self.size)] 
                                                              for i in range(self.size)])

        return board

    def get_next_play(self, board: Board) -> Play:
        x = self.translate_board(board.board)
        play = np.argmax(self.mcts.getActionProb(x, temp=0))
        return Play(row=play//self.size, col=play%self.size)

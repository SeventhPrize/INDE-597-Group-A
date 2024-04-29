import copy
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


class BoardException(Exception):
    pass


class InvalidMoveBoardException(BoardException):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        class_name = self.__class__.__name__
        return f"Invalid move ({self.x}, {self.y} ({class_name}))"


class CellNotEmpty(InvalidMoveBoardException):
    pass


class InmediateCapture(InvalidMoveBoardException):
    pass


class ProhibitionOfRepetition(InvalidMoveBoardException):
    pass


class CellDoesNotExist(InvalidMoveBoardException):
    pass


@dataclass(frozen=True)
class Play:
    row: int
    col: int


@dataclass(frozen=True)
class Skip(Play):
    row: int = -1
    col: int = -1


class Board:
    def __init__(self, size=5, board=None, first_pos=(0, 0)):
        if board:
            self.board = board
            self.size = len(board)
        else:
            self.size = size
            self.board = [["#" for j in range(self.size)] for i in range(self.size)]

        self.first_pos = first_pos

    def __getitem__(self, item):
        return self.board[item[0]][item[1]]

    def __setitem__(self, key, value):
        self.board[item[0]][item[1]] = value

    def __str__(self):
        return "\n".join([" ".join(row) for row in self.board])

    def perform_move(self, play: Play, color: str):
        if isinstance(play, Skip):
            return
        x = play.row
        y = play.col
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            raise CellDoesNotExist(x, y)
        if self.board[x][y] != "#":
            raise CellNotEmpty(x, y)
        previous_board = copy.deepcopy(self.board)
        self.board[x][y] = color
        try:
            self.check_inmediate_capture(x, y, color)
        except InmediateCapture:
            self.board = previous_board
            raise
        self.capture(color)
        if self.board == previous_board:
            self.board = previous_board
            raise ProhibitionOfRepetition(x, y)

    def capture(self, color):
        visited = set()
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] != "#" and self.board[i][j] != color and (i, j) not in visited:
                    visited.add((i, j))
                    group = self.get_group(i, j, self.board[i][j])
                    self.count_liberties(group)
                    if self.count_liberties(group) == 0:
                        for x, y in group:
                            self.board[x][y] = "#"

    def check_inmediate_capture(self, x, y, color):
        group = self.get_group(x, y, color)
        if self.count_liberties(group) == 0:
            raise InmediateCapture(x, y)

    def get_group(self, x, y, color):
        visited = set()
        to_visit = [(x, y)]
        while to_visit:
            x, y = to_visit.pop()
            visited.add((x, y))
            if x > 0 and self.board[x - 1][y] == color and (x - 1, y) not in visited:
                to_visit.append((x - 1, y))
            if x < self.size - 1 and self.board[x + 1][y] == color and (x + 1, y) not in visited:
                to_visit.append((x + 1, y))
            if y > 0 and self.board[x][y - 1] == color and (x, y - 1) not in visited:
                to_visit.append((x, y - 1))
            if y < self.size - 1 and self.board[x][y + 1] == color and (x, y + 1) not in visited:
                to_visit.append((x, y + 1))
        logger.info("Group", x=x, y=y, group=visited)
        return visited

    def count_liberties(self, group):
        liberties = set()
        for x, y in group:
            if x > 0 and self.board[x - 1][y] == "#" and (x - 1, y) not in liberties:
                liberties.add((x - 1, y))
            if x < self.size - 1 and self.board[x + 1][y] == "#" and (x + 1, y) not in liberties:
                liberties.add((x + 1, y))
            if y > 0 and self.board[x][y - 1] == "#" and (x, y - 1) not in liberties:
                liberties.add((x, y - 1))
            if y < self.size - 1 and self.board[x][y + 1] == "#" and (x, y + 1) not in liberties:
                liberties.add((x, y + 1))

        logger.info("Liberties", group=group, liberties=liberties, lcount=len(liberties))
        return len(liberties)

    def __eq__(self, other):
        return self.board == other.board

    def get_empty_cells(self):
        return [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i][j] == "#"]

    def get_hash(self):
        return hash(str(self.board))

    def get_winner(self):
        pass
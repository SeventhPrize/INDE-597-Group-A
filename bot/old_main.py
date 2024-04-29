from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import time
from selenium.common.exceptions import NoSuchElementException, WebDriverException
from PIL import Image
import copy
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


class Board:
    def __init__(self, size=6, board=None):
        if board:
            self.board = board
            self.size = len(board)
        else:
            self.size = size
            self.board = [["#" for j in range(self.size)] for i in range(self.size)]

    def __getitem__(self, item):
        return self.board[item[0]][item[1]]

    def __setitem__(self, key, value):
        self.board[item[0]][item[1]] = value

    def __str__(self):
        return "\n".join([" ".join(row) for row in self.board])

    def perform_move(self, x, y, color):
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

class BotPlayer:

    def __init__(self):
        firefox_options = Options()
        #firefox_options.add_argument('--headless')  # Run in headless mode

        #no use gpu accelerator
        firefox_options.add_argument('--disable-gpu')  # Disable GPU acceleration

        # firefox_options.set_preference("network.cookie.cookieBehavior", 2)  # Set to block all cookies by default

        # Create a web driver instance
        self.driver = webdriver.Firefox(options=firefox_options)

    def __del__(self):
        self.driver.quit()

    def get_board(self):
        field = smart_player.driver.find_element(By.XPATH, '//*[@id="stone"]')
        # take a screenshot
        field.screenshot('screenshot.png')
        # open image as a pixel array
        img = Image.open('screenshot.png')
        pixels = img.load()
        board = smart_player.parse_matrix(pixels)

        return Board(board=board)
    def parse_matrix(self, pixels):
        board = []
        for i in range(6):
            row = []
            for j in range(6):
                pixel = pixels[95 + j*100, 95 + i*100]
                #print(f"({i}, {j}) = ({95 + i*100}, {95 + j*100}) = {pixel}")
                if pixel == (42, 42, 42, 255):
                    row.append("#")
                elif pixel == (109, 109, 109, 255):
                    row.append("B")
                elif pixel == (255, 255, 255, 255):
                    row.append("W")
                elif pixel == (184, 164, 182, 255):
                    if pixels[95 + j*100, 95 + i*100 - 15] == (249, 249, 249, 255):
                        row.append("W")
                    else:
                        row.append("B")
            board.append(row)
        return board

    def click_pass_button(self):
        # click pass button
        logger.info("click pass button")
        field = smart_player.driver.find_element(By.XPATH, '//img[@id="pass-2"]')
        field.click()
        time.sleep(.5)
        field = smart_player.driver.find_element(By.XPATH, '//*[@id="yes"]')
        field.click()
        time.sleep(.5)
        logger.info("pass button clicked")

    def check_for_opponent_passing_its_turn(self):
        # check if the opponent is passing its turn
        logger.info("check if the opponent is passing its turn")
        # field with content "Pass"
        try:
            smart_player.driver.find_element(By.XPATH, '//*[text()="COSUMI Passed"]')
        except NoSuchElementException:
            return False
        try:
            smart_player.driver.find_element(By.XPATH, '//*[@id="thearea"]/div[@id="ok"]').click()
            time.sleep(.5)
            smart_player.driver.find_element(By.XPATH, '//*[@id="thearea"]/div[@id="ok"]').click()
        except Exception:
            pass
        else:
            time.sleep(1)
        return True

    def check_for_game_finished(self):
        logger.info("check for game finished")
        try:
            time.sleep(1)
            field = smart_player.driver.find_element(By.XPATH, '//div[@id="message"]')
            # get field text
            text = field.text
            logger.info("check for game finished", text=text)
            self.wait_for_browser_is_closed()
            return True
        except NoSuchElementException:
            return False

    def wait_for_browser_is_closed(self):
        # Now wait if someone closes the window
        logger.info("wait for browser is closed")
        while True:
            try:
                _ = self.driver.window_handles
            except WebDriverException as e:
                break
            time.sleep(1)


if __name__ == '__main__':
    try:
        # Create a SmartPlayer instance
        smart_player = BotPlayer()

        # Navigate to the URL
        smart_player.driver.get('https://playgo.to/en')
        # Wait for the page to load
        #time.sleep(1)
        # click element with id "choice-6"
        print("click element with id choice-6")
        
        # fix an error below
        iframe = smart_player.driver.find_element(By.TAG_NAME, "iframe")
        smart_player.driver.switch_to.frame(iframe)
        # Click element by css selector
        field = smart_player.driver.find_element(By.CSS_SELECTOR, '#choice-6')
        field.click()
        board = smart_player.get_board()
        while not smart_player.check_for_game_finished():
            print(board)
            x = input("x: ")
            y = input("y: ")
            if x == "pass":
                smart_player.click_pass_button()
            else:
                logger.info("Move to perform", x=x, y=y)
                try:
                    x, y = int(x), int(y)
                    board.perform_move(x, y, "B")
                    logger.info("Move performed", x=x, y=y)
                    print(board)

                    # click element with id "choice-6"
                    logger.info("Performing Moving on the board")
                    field = smart_player.driver.find_element(By.XPATH, '//*[@id="stone"]')
                    action = webdriver.common.action_chains.ActionChains(smart_player.driver)
                    action.move_to_element_with_offset(field, -255 + y * 95, -255 + x * 95).click().perform()
                    logger.info("Move Performed")
                    time.sleep(1)
                except InvalidMoveBoardException as e:
                    logger.exception("Move not performed", x=x, y=y, error=e)
                    continue

            new_board = smart_player.get_board()
            while board == new_board:
                logger.info("Waiting for the opponent to move")
                time.sleep(1)
                if smart_player.check_for_opponent_passing_its_turn():
                    logger.info("Opponent passed its turn")
                    break
                new_board = smart_player.get_board()
            else:
                logger.info("Opponent moved")
                board = new_board
    #    except Exception as e:

    finally:
        # close the browser
        smart_player.driver.quit()

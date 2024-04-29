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
from board import Board, InvalidMoveBoardException, Play, Skip
from player import Player, HumanPlayer, RandomPlayer, ChatGPTPlayer, RLPlayer, MCTSPlayer
import typer
from rich.console import Console
from enum import Enum

app = typer.Typer(
    name="Price Tag Generator",
    help="Generate Price Tags",
    add_completion=False,
)

logger = structlog.get_logger(__name__)
console = Console()


class BotPlayer:

    def __init__(self, player: Player, size=5):
        logger.info("Initializing BotPlayer", player=player, size=size)

        firefox_options = Options()
        #firefox_options.add_argument('--headless')  # Run in headless mode

        #no use gpu accelerator
        firefox_options.add_argument('--disable-gpu')  # Disable GPU acceleration

        # firefox_options.set_preference("network.cookie.cookieBehavior", 2)  # Set to block all cookies by default

        # Create a web driver instance
        self.driver = webdriver.Firefox(options=firefox_options)
        self.player = player
        self.size = size
        self.winner = None
        self.first_pos = None
        

    def __del__(self):
        self.driver.quit()

    def get_board(self):
        field = self.driver.find_element(By.XPATH, '//*[@id="stone"]')
        # take a screenshot
        field.screenshot('screenshot.png')

        # open image as a pixel array
        img = Image.open('screenshot.png')
        pixels = img.load()

        if self.first_pos is None:
            logger.info("Getting first position")
            # search for first black pixel from the top left corner
            first_black_pixel = next((i,j) for i in range(img.size[0]) 
                                    for j in range(img.size[1]) if pixels[i, j] == (42, 42, 42, 255))

            first_black_pixel = (first_black_pixel[0] + 1, first_black_pixel[1])
            self.first_pos = first_black_pixel
            self.block_size = 100 if self.size < 9 else 76
        
        board = self.parse_matrix(pixels)

        return Board(board=board, size=self.size, first_pos=self.first_pos)
    
    def parse_matrix(self, pixels):
        logger.info("First black pixel", first_black_pixel=self.first_pos)
        board = []
        for i in range(self.size):
            row = []
            for j in range(self.size):
                logger.info("Parsing matrix", i=i, j=j, x=self.first_pos[0] + j*self.block_size, y=self.first_pos[1] + i*self.block_size)
                pixel = pixels[self.first_pos[0] + j*self.block_size, self.first_pos[1] + i*self.block_size]
                #print(f"({i}, {j}) = ({95 + i*100}, {95 + j*100}) = {pixel}")
                if pixel == (42, 42, 42, 255):
                    row.append("#")
                elif pixel == (109, 109, 109, 255):
                    row.append("B")
                elif pixel == (255, 255, 255, 255):
                    row.append("W")
                elif pixel == (184, 164, 182, 255):
                    if pixels[self.first_pos[0] + j*self.block_size, self.first_pos[1] + i*self.block_size - 15] == (249, 249, 249, 255):
                        row.append("W")
                    else:
                        row.append("B")
            board.append(row)
        return board

    def click_pass_button(self):
        # click pass button
        logger.info("click pass button")
        field = self.driver.find_element(By.XPATH, '//div[@id="passbutton"]')
        action = webdriver.common.action_chains.ActionChains(self.driver)
        action.move_to_element_with_offset(field, 0,0).click().perform()
        time.sleep(1)
        field = self.driver.find_element(By.XPATH, '//div[@id="yes"]')
        action = webdriver.common.action_chains.ActionChains(self.driver)
        action.move_to_element_with_offset(field, 0,0).click().perform()
        time.sleep(1)
        logger.info("pass button clicked")

    def check_for_opponent_passing_its_turn(self):
        # check if the opponent is passing its turn
        logger.info("check if the opponent is passing its turn")
        # field with content "Pass"
        try:
            field = self.driver.find_element(By.XPATH, '//div[@id="message"]')
            text = field.text
            logger.info("check for opponent passing its turn", text=text)
            if "cosumi passed" not in text.lower():
                return False
            #self.driver.find_element(By.XPATH, '//*[text()="COSUMI Passed"]')
        except NoSuchElementException:
            return False
        try:
            logger.info("Cossumi passed")
            self.driver.find_element(By.XPATH, '//*[@id="thearea"]/div[@id="ok"]').click()
            time.sleep(.5)
            self.driver.find_element(By.XPATH, '//*[@id="thearea"]/div[@id="ok"]').click()
        except Exception:
            pass
        else:
            time.sleep(1)
        return True

    def check_for_game_finished(self):
        logger.info("check for game finished")
        try:
            time.sleep(1)
            field = self.driver.find_element(By.XPATH, '//div[@id="message"]')
            # get field text
            text = field.text
            logger.info("check for game finished", text=text)
            if "win" in text.lower() or "lose" in text.lower():
                self.winner = 'B' if "win" in text.lower() else 'W'
                self.wait_for_browser_is_closed()
                return True
            return False
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

    def play(self):
        # Navigate to the URL
        #self.driver.get('https://playgo.to/en')
        self.driver.get('https://www.cosumi.net/en/')
        # Wait for the page to load
        #time.sleep(1)
        # click element with id "choice-{self.size}"
        
        # fix an error below
        #iframe = self.driver.find_element(By.TAG_NAME, "iframe")
        #self.driver.switch_to.frame(iframe)
        # Click element by css selector
        
        selector = f'#choice-{self.size}'
        logger.info("click element", selector=selector)
        field = self.driver.find_element(By.CSS_SELECTOR, selector)
        field.click()
        breakpoint()
        
        board = self.get_board()
        while not self.winner:
            print(board)

            play = self.player.get_next_play(board)
            old_board = copy.deepcopy(board)

            if isinstance(play, Skip):
                self.click_pass_button()
            else:
                logger.info("Move to perform", play=play)
                try:
                    board.perform_move(play, "B")
                except InvalidMoveBoardException as e:
                    logger.error("Invalid move", error=e)
                    self.click_pass_button()

                logger.info("Performing Moving on the board")

                field = self.driver.find_element(By.XPATH, '//*[@id="stone"]')
                action = webdriver.common.action_chains.ActionChains(self.driver)
                #visualize where the click will be performed
                
                #action.move_to_element_with_offset(field, -255 + play.col * 95, -255 + play.row * 95).context_click().perform()


                pos_to_click =  (field.size['height']//-2 + board.first_pos[0] + play.col * self.block_size, field.size['width']//-2 + board.first_pos[0] + play.row * self.block_size) 
                action.move_to_element_with_offset(field, *pos_to_click).click().perform()
                
    
                logger.info("Move performed", play=play)
                print(board)
                time.sleep(1)


            attemps = 0
            new_board = self.get_board()
            while board == new_board and attemps<=10:
                logger.info("Waiting for the opponent to move", attemps=attemps)
                time.sleep(1)
                if self.check_for_opponent_passing_its_turn():
                    logger.info("Opponent passed its turn")
                    break
                new_board = self.get_board()
                attemps += 1
            else:
                if self.check_for_opponent_passing_its_turn():
                    time.sleep(1)
                    new_board = self.get_board()
                logger.info("Opponent moved")
                board = new_board
                self.check_for_game_finished()
                self.player.update_reward(old_board, play)

AVAILABLE_PLAYERS = {
    "RandomPlayer": RandomPlayer,
    "HumanPlayer": HumanPlayer,
    "ChatGPTPlayer": ChatGPTPlayer,
    "RLPlayer": RLPlayer,
    "MCTSPlayer": MCTSPlayer,
}


class Player(str, Enum):
    RandomPlayer = "RandomPlayer"
    HumanPlayer = "HumanPlayer"
    ChatGPTPlayer = "ChatGPTPlayer"
    RLPlayer = "RLPlayer"
    MCTSPlayer = "MCTSPlayer"


@app.command()
def play_go(
        player: Player = typer.Option(case_sensitive=False, help="Player to use"),
        board_size: int = typer.Option(5, help="Board size"),
):
    player_cls = AVAILABLE_PLAYERS[player]
    bot = BotPlayer(player_cls("B", size=board_size), size=board_size)
    bot.play()


if __name__ == '__main__':
    app()
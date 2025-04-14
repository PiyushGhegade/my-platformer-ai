import pygame, sys
from settings import screen_width, screen_height
from level import Level
from ui import UI
import random

class Game:
    def __init__(self, external_screen=None):
        """Initialize the game. Accepts an external screen if running from Gym."""
        self.max_health = 100
        self.cur_health = 100
        self.coins = 0
        self.cur_level = 1  # Randomly select a level between 1 and 5
        # Audio 
        self.level_bg_music = pygame.mixer.Sound('../audio/level_music.wav')

        # Use an external screen if provided (Gym), otherwise create a new one
        self.screen = external_screen if external_screen else pygame.display.set_mode((screen_width, screen_height))
        
        # Directly start cur_level
        self.level = Level(self.cur_level, self.screen, self.change_coins, self.change_health)
        self.status = 'level'
        self.level_bg_music.play(loops=-1)

        # UI setup
        self.ui = UI(self.screen)

    def change_coins(self, amount):
        self.coins += amount

    def change_health(self, amount):
        self.cur_health += amount

    def reset(self):
        """Restart the level instead of quitting the game"""
        self.cur_health = 100  # Reset health
        self.coins = 0  # Reset coins
        self.level = Level(self.cur_level, self.screen, self.change_coins, self.change_health)  # Restart cur_level
        self.status = 'level'

    def run(self):
        """Run one frame of the game"""
        self.level.run()
        self.ui.show_health(self.cur_health, self.max_health)
        self.ui.show_coins(self.coins)

        # Restart the level when health reaches zero
        if self.cur_health <= 0:
            self.reset()
        # Give condition to reset if player fall

# Run the game normally if executed directly
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    game = Game(screen)

    #Main Code
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill('grey')
        game.run()

        pygame.display.update()
        clock.tick(60)

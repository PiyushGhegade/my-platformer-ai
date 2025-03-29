import pygame, sys
from settings import * 
from level import Level
from ui import UI



class Game:
    def __init__(self, external_screen=None):
        """Initialize the game. Accepts an external screen if running from Gym."""
        self.max_health = 100
        self.cur_health = 100
        self.coins = 0
        
        # Audio 
        self.level_bg_music = pygame.mixer.Sound('../audio/level_music.wav')

        # Use an external screen if provided (Gym), otherwise create a new one
        self.screen = external_screen if external_screen else pygame.display.set_mode((screen_width, screen_height))
        
        # Directly start Level 1
        self.level = Level(0, self.screen, self.change_coins, self.change_health)
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
        self.level = Level(0, self.screen, self.change_coins, self.change_health)  # Restart Level 1
        self.status = 'level'

    def run(self):
        """Run one frame of the game"""
        self.level.run()
        self.ui.show_health(self.cur_health, self.max_health)
        self.ui.show_coins(self.coins)

        # Restart the level when health reaches zero
        if self.cur_health <= 0:
            self.reset()

# Run the game normally if executed directly
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    game = Game(screen)

    csv_file = "../levels/2/level_2_terrain.csv"
    set_1 = {0, 3, 12, 15}
    set_2 = {2, 3, 14, 15}
    list_1, list_2 = extract_cell_positions(csv_file, set_1, set_2)
    print(list_1)

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

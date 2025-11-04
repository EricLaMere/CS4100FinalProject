import vis_2048
import sys
import pygame
from game_2048 import Game2048 

# Constants
WINDOW_SIZE = 500
GRID_SIZE = 4
CELL_SIZE = WINDOW_SIZE // (GRID_SIZE + 1)
CELL_PAD = 10
FPS = 60

# Colors
BACKGROUND = (187, 173, 160)
EMPTY_CELL = (205, 193, 180)
CELL_COLORS = {
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
    4096: (60, 58, 50),
}
TEXT_DARK = (119, 110, 101)
TEXT_LIGHT = (249, 246, 242)

# Initialize Pygame
pygame.init()

# Setup display
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption('2048')
clock = pygame.time.Clock()

def draw_grid(game):
    screen.fill(BACKGROUND)

    # Draw title and score
    title_font = pygame.font.Font(None, 60)
    score_font = pygame.font.Font(None, 36)

    title = title_font.render('2048', True, TEXT_DARK)
    screen.blit(title, (20, 20))

    score_text = score_font.render(f'Score: {game.score}', True, TEXT_DARK)
    screen.blit(score_text, (WINDOW_SIZE - 180, 30))

    # Draw grid
    offset = 80
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            value = game.grid[i][j]
            x = j * CELL_SIZE + CELL_PAD + offset
            y = i * CELL_SIZE + CELL_PAD + offset

            # Draw cell background
            color = CELL_COLORS.get(value, CELL_COLORS[4096]) if value else EMPTY_CELL
            pygame.draw.rect(screen, color, (x, y, CELL_SIZE - 2 * CELL_PAD, CELL_SIZE - 2 * CELL_PAD), border_radius=5)

            # Draw number
            if value:
                font_size = 55 if value < 100 else (45 if value < 1000 else 35)
                font = pygame.font.Font(None, font_size)
                text_color = TEXT_DARK if value <= 4 else TEXT_LIGHT
                text = font.render(str(value), True, text_color)
                text_rect = text.get_rect(
                    center=(x + (CELL_SIZE - 2 * CELL_PAD) // 2, y + (CELL_SIZE - 2 * CELL_PAD) // 2))
                screen.blit(text, text_rect)

    # Game over message
    if game.game_over:
        overlay = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
        overlay.set_alpha(180)
        overlay.fill((255, 255, 255))
        screen.blit(overlay, (0, 0))

        game_over_font = pygame.font.Font(None, 72)
        restart_font = pygame.font.Font(None, 36)

        game_over_text = game_over_font.render('Game Over!', True, TEXT_DARK)
        restart_text = restart_font.render('Press R to Restart', True, TEXT_DARK)

        screen.blit(game_over_text, (WINDOW_SIZE // 2 - game_over_text.get_width() // 2, WINDOW_SIZE // 2 - 50))
        screen.blit(restart_text, (WINDOW_SIZE // 2 - restart_text.get_width() // 2, WINDOW_SIZE // 2 + 20))

    pygame.display.flip()


def main():
    game = Game2048(GRID_SIZE)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game.reset()
                elif event.key == pygame.K_LEFT:
                    game.move('LEFT')
                elif event.key == pygame.K_RIGHT:
                    game.move('RIGHT')
                elif event.key == pygame.K_UP:
                    game.move('UP')
                elif event.key == pygame.K_DOWN:
                    game.move('DOWN')

        draw_grid(game)
        clock.tick(FPS)


if __name__ == '__main__':
    main()
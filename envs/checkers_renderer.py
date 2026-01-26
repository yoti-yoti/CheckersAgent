# envs/checkers_renderer.py
import pygame
import numpy as np

class CheckersRenderer:
    def __init__(self, cell_size=80, fps=30, title="Checkers"):
        pygame.init()
        self.cell_size = cell_size
        self.fps = fps
        self.clock = pygame.time.Clock()
        self.size = 8 * cell_size
        self.screen = pygame.display.set_mode((self.size, self.size + 60))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("Arial", 22)

    def close(self):
        pygame.quit()

    def pump(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit

    def draw(self, board: np.ndarray, hud=None):
        self.pump()
        cs = self.cell_size

        light = (238, 238, 210)
        dark = (118, 150, 86)
        p1 = (220, 20, 60)
        p2 = (25, 25, 25)
        crown = (255, 215, 0)

        hud_bg = (30, 30, 30)
        hud_fg = (235, 235, 235)

        self.screen.fill((0, 0, 0))

        for r in range(8):
            for c in range(8):
                color = dark if (r + c) % 2 == 1 else light
                pygame.draw.rect(self.screen, color, pygame.Rect(c * cs, r * cs, cs, cs))

                v = int(board[r, c])
                if v == 0:
                    continue

                center = (c * cs + cs // 2, r * cs + cs // 2)
                radius = int(cs * 0.38)
                piece_color = p1 if v > 0 else p2
                pygame.draw.circle(self.screen, piece_color, center, radius)

                if abs(v) == 2:
                    pygame.draw.circle(self.screen, crown, center, int(radius * 0.45), width=6)

        pygame.draw.rect(self.screen, hud_bg, pygame.Rect(0, 8 * cs, self.size, 60))

        if hud is None:
            hud = {}
        text = " | ".join([f"{k}: {v}" for k, v in hud.items()])

        surf = self.font.render(text, True, hud_fg)
        self.screen.blit(surf, (12, 8 * cs + 18))

        pygame.display.flip()
        self.clock.tick(self.fps)

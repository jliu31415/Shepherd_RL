import pygame
import numpy as np

class Unit:
    def __init__(self, init_pos, color):
        self.pos = np.array(init_pos)
        self.color = color

    def render(self, screen):
        pygame.draw.circle(screen, self.color, self.pos, 1, 0)

    def get_pos(self):
        return self.pos

    def update_pos(self, increment):
        self.pos = np.add(self.pos, increment)
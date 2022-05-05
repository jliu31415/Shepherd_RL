import numpy as np
from parameters import *
from unit import Unit

class Shepherd(Unit):
    def __init__(self, init_pos):
        super().__init__(init_pos, (255, 0, 0))

    def step(self, angle, scale=1):
        increment = [delta_s*scale*np.cos(angle), delta_s*scale*np.sin(angle)]
        self.update_pos(increment)

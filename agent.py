import numpy as np
from helpers import *
from parameters import *
from unit import Unit
from functools import reduce

class Agent(Unit):
    def __init__(self, init_pos):
        super().__init__(init_pos, (0, 255, 0))
        self.h_vect = 0

    def step(self, i, shepherd, agent_pos, distances):
        # local repulsion
        R_ai = [unit_vect(agent_pos[i], agent_pos[j]) for j in range(0, num_agents) 
                    if i != j and distances[min(i,j)][abs(i-j)-1] < r_a]
        R_a = 0 if len(R_ai) == 0 else unit_vect(reduce(np.add, R_ai))
        
        if dist(agent_pos[i], shepherd.get_pos()) > r_s:
            # if shepherd not in range, update with local repulsion and graze
            self.h_vect = p_a*R_a
            if np.random.rand() < graze_prob:
                self.h_vect += rand_unit()         
        else:
            # attracted to local center of mass of nearest agents (ignore self at index 0)
            c_vect = 0
            if num_nearest > 0:
                nearest_positions = sorted(agent_pos, key=lambda x: dist(agent_pos[i], x))[1:num_nearest+1]
                local_center = reduce(np.add, nearest_positions)/num_nearest
                c_vect = unit_vect(local_center, agent_pos[i])
            
            # repelled from shepherd
            R_s = unit_vect(agent_pos[i], shepherd.get_pos())
            self.h_vect = h*unit_vect(self.h_vect) + c*c_vect + p_a*R_a + p_s*R_s + e*rand_unit()
            
        self.update_pos(delta_a*unit_vect(self.h_vect))
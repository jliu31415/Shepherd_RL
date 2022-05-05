import pygame
import numpy as np
from helpers import *
from parameters import *
from shepherd import Shepherd
from agent import Agent
from functools import reduce

FPS = 1200
fpsClock = pygame.time.Clock()

class Environment:
    def __init__(self):
        self.width = 200
        self.height = self.width
        self.padding = 30
        # screen dimensions, padding included
        self.screen_width = self.width+2*self.padding
        self.screen_height = self.height+2*self.padding
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.target = [self.width+self.padding, self.height+self.padding]       # target location
        self.target_radius = 20     # distance from target to trigger win condition
        # shepherd and agents initalized in reset function
        self.shepherd = None
        self.agents = []
        self.agent_posns = []
        self.reset()

    def reset(self):
        # start shepherd at target, agents in upper left hand corner
        # self.shepherd = Shepherd(self.target)
        # self.agents = [Agent([self.padding+L*np.random.rand(), self.padding+L*np.random.rand()]) 
        #                 for i in range(0, num_agents)]
        # start shepherd behind agents
        self.shepherd = Shepherd([self.padding, self.padding])
        offset = self.padding+r_s/1.4
        self.agents = [Agent([offset+init_field*np.random.rand(), offset+init_field*np.random.rand()]) 
                        for i in range(0, num_agents)]
        self.agent_posns = [a.get_pos() for a in self.agents]

    def step(self, action=None):
        # update shepherd; action is None --> mouse input
        if action is None:
            control_vect = np.subtract(pygame.mouse.get_pos(), self.shepherd.get_pos())
            angle = np.arctan(control_vect[1]/control_vect[0])
            if control_vect[0] < 0:
                angle += np.pi
            scale = min(dist(control_vect), 50)/50
            self.shepherd.step(angle, scale)
        else:
            if action == 0:
                self.shepherd.step(0, 0)
            else:
                self.shepherd.step(np.pi*(action % 4)/2, 1)

        # distances[i][j-i-1] will contain distance from i to j, i < j
        distances = [[dist(self.agent_posns[i], self.agent_posns[j]) for j in range(i+1, num_agents)] 
                    for i in range(0, num_agents-1)]
        [a.step(i, self.shepherd, self.agent_posns, distances) for i, a in enumerate(self.agents)]
        # update agent positions
        self.agent_posns = [a.get_pos() for a in self.agents]

        # give a reward based on GCM distance to target
        gcm = reduce(np.add, self.agent_posns)/len(self.agent_posns)
        gcm_to_target = dist(self.target, gcm)
        reward = -gcm_to_target/(self.width*1.41)

        shep_to_agent = min([dist(agent_pos, self.shepherd.get_pos()) for agent_pos in self.agent_posns])
        reward -= (shep_to_agent/r_s - 1)

        # penalize if shepherd goes off screen
        # if self.shepherd.get_pos()[0] < 0  or self.shepherd.get_pos()[0] > self.screen_width:
        #     if self.shepherd.get_pos()[1] < 0 or self.shepherd.get_pos()[1] > self.screen_height:
        #         reward -= 1

        # # testing purposes: objective shepherd to target
        # reward = -dist(self.shepherd.get_pos(), self.target)
        # if -reward < self.target_radius:
        #     return 1000, True
        # return reward, False

        # return game_won = True if furthest agent within target_radius
        game_over = False
        max_agent_dist = max([dist(agent_pos, self.target) for agent_pos in self.agent_posns])
        if max_agent_dist < self.target_radius:
            reward = 1000
            game_over = True
            
        return reward, game_over

    def render(self):
        self.screen.fill((0, 0, 0))
        pygame.draw.circle(self.screen, (255, 255, 255), self.target, 1, 0)
        self.shepherd.render(self.screen)
        [a.render(self.screen) for a in self.agents]
        pygame.display.update()

    def get_state(self):
        arr1 = np.array(self.agent_posns).flatten()
        arr2 = np.array(self.shepherd.get_pos()).flatten()
        return np.concatenate((arr1, arr2), axis=0)

    def pygame_running(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def run(self, dqn_agent=None):        
        # game loop
        while self.pygame_running():
            self.render()
            if dqn_agent is None:
                game_over = self.step()[1]
            else:
                action = dqn_agent.get_action(self.get_state())
                game_over = self.step(action)[1]
            if game_over:
                self.reset()
            fpsClock.tick(FPS)

pygame.init()
if __name__ == "__main__":
    Environment().run()
    
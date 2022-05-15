import pygame
import numpy as np
from helpers import *
from parameters import *
from shepherd import Shepherd
from agent import Agent
from functools import reduce

FPS = 80
fpsClock = pygame.time.Clock()

class Environment:
    def __init__(self, render):
        self.padding = 35
        if render:
            pygame.init()
            screen_length = field_length+2*self.padding
            self.screen = pygame.display.set_mode((screen_length, screen_length))
        # initalized in reset function
        self.shepherd = None
        self.agents = []
        self.agent_posns = []
        self.target = []    # target location
        self.key_action = 0
        self.reset()

    def reset(self):
        # self.init_v1()        
        self.init_v2()
        # self.init_v3()
        self.agent_posns = [a.get_pos() for a in self.agents]

    def init_v1(self):
        # start shepherd behind agents
        self.target = [field_length+self.padding, field_length+self.padding]
        self.shepherd = Shepherd([self.padding, self.padding])
        offset = self.padding+r_s/1.4
        self.agents = []
        for _ in range(0, num_agents):
            a = Agent([offset+agent_init*np.random.rand(), 
                        offset+agent_init*np.random.rand()]) 
            self.agents.append(a)
    
    def init_v2(self):
        # start shepherd at target, agents in upper left hand corner
        self.target = [field_length+self.padding, field_length+self.padding]
        self.shepherd = Shepherd(self.target)
        self.agents = []
        for _ in range(0, num_agents):
            a = Agent([self.padding+agent_init*np.random.rand(), 
                        self.padding+agent_init*np.random.rand()]) 
            self.agents.append(a)

    def init_v3(self):
        # randomize shepherd and agent positions, set target to center
        self.target = [field_length/2+self.padding, field_length/2+self.padding]
        self.shepherd = Shepherd([self.padding+field_length*np.random.rand(),
                                    self.padding+field_length*np.random.rand()])
        self.agents = []
        for _ in range(0, num_agents):
            a = Agent([self.padding+field_length*np.random.rand(), 
                        self.padding+field_length*np.random.rand()]) 
            self.agents.append(a)
        pass

    def step(self, action=None):
        # update shepherd; action is None --> mouse input
        if action is None:
            # choose between mouse or keyboard input
            self.step_mouse_input()
            # self.step_key_input()           
        else:
            self.shepherd.step(np.pi*(action % 4)/2, 1)

        # distances[i][j-i-1] will contain distance from i to j, i < j
        distances = [[dist(self.agent_posns[i], self.agent_posns[j]) for j in range(i+1, num_agents)] 
                    for i in range(0, num_agents-1)]
        [a.step(i, self.shepherd, self.agent_posns, distances) for i, a in enumerate(self.agents)]
        # update agent positions
        self.agent_posns = [a.get_pos() for a in self.agents]

        # return self.test_dqn()

        # give a reward based on GCM distance to target
        gcm = reduce(np.add, self.agent_posns)/len(self.agent_posns)
        gcm_to_target = dist(self.target, gcm)
        reward = -(gcm_to_target-target_radius)/(field_length*1.41)
        
        # encourage shepherd to approach agent
        shep_to_agent = min([dist(agent_pos, self.shepherd.get_pos()) for agent_pos in self.agent_posns])
        reward -= shep_to_agent/r_s - 1
        
        # return game_won = True if furthest agent within target_radius
        game_over = False
        max_agent_dist = max([dist(agent_pos, self.target) for agent_pos in self.agent_posns])
        if max_agent_dist < target_radius:
            reward = 1000
            game_over = True
            
        return reward, game_over

    def test_dqn(self):
        distance = dist(self.shepherd.get_pos(), self.target)
        if (distance < 20):
            return 1000, True
        return -distance, False

    def step_mouse_input(self):
        control_vect = np.subtract(pygame.mouse.get_pos(), self.shepherd.get_pos())
        angle = np.arctan(control_vect[1]/control_vect[0])
        if control_vect[0] < 0:
            angle += np.pi
        scale = min(dist(control_vect), 50)/50
        self.shepherd.step(angle, scale)

    def step_key_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT]:
            self.key_action = 0
        elif keys[pygame.K_DOWN]:
            self.key_action = 1
        elif keys[pygame.K_LEFT]:
            self.key_action = 2
        elif keys[pygame.K_UP]:
            self.key_action = 3
        self.shepherd.step(np.pi*(self.key_action % 4)/2, 1)

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
            keys = pygame.key.get_pressed()
            if game_over or keys[pygame.K_r]:
                self.reset()
            fpsClock.tick(FPS)

if __name__ == "__main__":
    Environment(True).run()
    
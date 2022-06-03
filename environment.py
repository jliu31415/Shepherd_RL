import pygame
import numpy as np
from parameters import *
from functools import reduce
import threading

FPS = 80
fpsClock = pygame.time.Clock()

class Environment:
    def __init__(self, render):
        self.padding = np.array([30, 30])   # padding for GUI
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((FIELD_LENGTH+2*self.padding[0],
                                                    FIELD_LENGTH+2*self.padding[1]))
        self.action = 0
        self.reset_enabled = True
        # initalized in reset function
        self.shepherd = None
        self.target = None
        self.agents = None
        self.num_agents = 0
        self.num_nearest = 0
        self.reset()

    def reset(self):
        self.num_agents = np.random.randint(1, MAX_NUM_AGENTS+1)
        self.num_nearest = self.num_agents-1
        # shepherd and target start at same random location
        # self.shepherd = np.random.randint(FIELD_LENGTH, size=2)
        # self.target = np.copy(self.shepherd)
        self.shepherd = np.array([0, 0])
        self.target = np.array([FIELD_LENGTH-1, FIELD_LENGTH-1])
        self.agents = R_S//2 + np.random.randint(FIELD_LENGTH/2, size=(self.num_agents, 2))

    def step(self, action):
        # map action [0, 3] to direction vector
        shep_direction = np.array([(1-action%2)*(-action+1), 
                                    (action%2)*(-action+2)])
        
        # update shepherd
        self.shepherd += shep_direction
        self.shepherd = self.shepherd.clip(0, FIELD_LENGTH-1)

        # update agents
        # lookup table distances[i][j] = dist(self.agent[i], self.agent[j])
        # if i >= j, distances[i][j] = 0
        distances = [[self.dist(self.agents[i], self.agents[j]) if i < j else 0 for j in range(self.num_agents)] 
                    for i in range(self.num_agents)]
        for i in range(self.num_agents):
            # local repulsion
            v_a_ = [self.unit_vect(self.agents[i], self.agents[j]) for j in range(self.num_agents) 
                            if i != j and distances[min(i,j)][max(i,j)] < R_A]
            # if two agents in same location, unit_vect returns zero; need to map to random unit vector
            for v_index in range(len(v_a_)):
                if (v_a_[v_index]==0).all():
                    v_a_[v_index] = self.rand_unit()
            v_a = 0 if len(v_a_) == 0 else self.unit_vect(reduce(np.add, v_a_))

            if (self.dist(self.shepherd, self.agents[i]) > R_S):
                # agent outside of shepherd detection radius, only consider local repulsion
                self.agents[i] = np.rint(self.agents[i] + self.unit_vect(v_a))
            else:
                # attracted to local center of mass of nearest agents (ignore self at index 0)
                v_c = 0
                if self.num_nearest > 0:
                    sorted_agents = sorted(self.agents, key=lambda x: self.dist(self.agents[i], x))
                    nearest_agents = sorted_agents[1:self.num_nearest+1]     # ignore self at index 0
                    lcm = reduce(np.add, nearest_agents)/self.num_nearest
                    v_c = self.unit_vect(lcm, self.agents[i])
                
                # repelled from shepherd (if in same location, run towards center of board)
                v_s = self.unit_vect(self.agents[i], self.shepherd)
                if (v_s==0).all():
                    v_s = self.unit_vect(np.array([FIELD_LENGTH/2, FIELD_LENGTH/2]), self.agents[i])

                self.agents[i] = np.rint(self.agents[i] + self.unit_vect(P_A*v_a + P_C*v_c + P_S*v_s))
        
        self.agents = self.agents.clip(0, FIELD_LENGTH-1)

        # return self.test_dqn()

        # give a reward based on GCM distance to target
        gcm = reduce(np.add, self.agents)/self.num_agents
        gcm_to_target = self.dist(self.target, gcm)
        reward = -(gcm_to_target-TARGET_RADIUS)/R_S
        
        # encourage shepherd to approach agent
        shep_to_agent = min([self.dist(a, self.shepherd) for a in self.agents])
        reward -= shep_to_agent/R_S - 1
        
        # return game_won = True if furthest agent within target_radius
        game_over = False
        max_agent_dist = max([self.dist(a, self.target) for a in self.agents])
        if max_agent_dist < TARGET_RADIUS:
            reward = 1000
            game_over = True
            
        return reward, game_over

    def test_dqn(self):
        pass

    # distance from [a1, a2] to [b1, b2]
    def dist(self, a, b=np.array([0, 0])):
        return np.linalg.norm(a-b)

    # unit vect from [b1, b2] to [a1, a2]
    def unit_vect(self, a, b=np.array([0, 0])):
        if self.dist(a, b) == 0:
            return np.array([0, 0])
        return (a-b)/self.dist(a, b)

    # return random unit vector
    def rand_unit(self):
        return self.unit_vect([np.random.rand()-.5, np.random.rand()-.5])

    def get_key_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT]:
            return 0
        elif keys[pygame.K_DOWN]:
            return 1
        elif keys[pygame.K_LEFT]:
            return 2
        elif keys[pygame.K_UP]:
            return 3
        return self.action

    def get_mouse_input(self):
        mouse_vect = pygame.mouse.get_pos() - self.padding - self.shepherd
        if abs(mouse_vect[0]) > abs(mouse_vect[1]):
            if mouse_vect[0] > 0:
                return 0
            return 2
        if mouse_vect[1] > 0:
            return 1
        return 3        

    def render(self):
        self.screen.fill((0, 0, 0))
        pygame.draw.circle(self.screen, (255, 0, 0), tuple(self.padding + self.shepherd), 1, 0)
        pygame.draw.circle(self.screen, (0, 255, 255), tuple(self.padding + self.target), 1, 0)
        [pygame.draw.circle(self.screen, (0, 255, 0), tuple(self.padding + a), 1, 0)
            for a in self.agents]
        pygame.display.update()

    def get_state(self):
        if CNN_NETWORK:
            state = np.zeros((FIELD_LENGTH, FIELD_LENGTH), dtype=np.int32)
            # 100 := shepherd
            # 200 := target
            # x % 100 := number of sheep at given coordinates
            state[self.shepherd[0], self.shepherd[1]] += 100
            state[self.target[0], self.target[1]] += 200
            for i, j in self.agents:
                state[i, j] += 1
            # add channel dimension
            return np.expand_dims(state, axis=0)
        else:
            # pad self.agents to maintain constant dimensions
            padded_agents = self.agents.flatten()
            padded_agents = np.pad(padded_agents, (0, 2*MAX_NUM_AGENTS-len(padded_agents)), 
                                'constant', constant_values=-100)
            return np.concatenate((padded_agents, self.shepherd.flatten(), self.target.flatten()), axis=0)

    def pygame_running(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def enable_reset(self):
        self.reset_enabled = True

    def run(self, dqn_agent=None):        
        # game loop
        while self.pygame_running():
            self.render()

            if dqn_agent is None:
                self.action = self.get_key_input()
                # self.action = self.get_mouse_input()
            else:
                self.action = dqn_agent.get_action(self.get_state())
            _, game_over = self.step(self.action)
            
            keys = pygame.key.get_pressed()
            if game_over or (keys[pygame.K_r] and self.reset_enabled):
                self.reset()
                self.reset_enabled = False
                threading.Timer(.5, self.enable_reset).start()
            if keys[pygame.K_ESCAPE]:
                break
            fpsClock.tick(FPS)

if __name__ == "__main__":
    Environment(True).run()
    
import numpy as np
from collections import deque
import random
from environment import Environment
from parameters import num_agents
from dqn import DQN
import torch as T
import time
import argparse

MODEL_PATH = 'model.pth'
AUTO_RESET = 500   # automatically reset game after x frames

class DQNAgent:
    def __init__(self, gamma, epsilon, lr, batch_size,
                    max_mem_size=100000, eps_min=0.01, eps_dec=5e-4):
        self.episode_num = 0
        self.gamma = gamma
        self.epsilon = epsilon        # randomness
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.lr = lr      # learning rate
        self.mem_size = max_mem_size
        self.memory = deque(maxlen=self.mem_size)
        self.input_size = 2*(num_agents+1)
        self.output_size = 5
        self.batch_size = batch_size
        self.dqn = DQN(self.lr, self.input_size, self.output_size)

    def remember(self, state_old, action, reward, state_new, game_over):
        self.memory.append((state_old, action, reward, state_new, game_over))

    def get_action(self, state):
        # exploration vs exploitation
        if random.random() < self.epsilon:
            action = random.randint(0, self.output_size-1)
        else:
            state_tensor = T.tensor(np.array(state), dtype=T.float).to(self.dqn.device)
            prediction = self.dqn.forward(state_tensor)
            action = T.argmax(prediction).item()
        return action

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        state_old, action, reward, state_new, game_over = zip(*batch)
        state_old = T.tensor(np.array(state_old), dtype=T.float).to(self.dqn.device)
        action = T.tensor(np.array(action), dtype=T.long).to(self.dqn.device)
        reward = T.tensor(np.array(reward), dtype=T.float).to(self.dqn.device)
        state_new = T.tensor(np.array(state_new), dtype=T.float).to(self.dqn.device)
        game_over = T.tensor(np.array(game_over), dtype=T.bool).to(self.dqn.device)
        
        self.dqn.optimizer.zero_grad()
        q = self.dqn.forward(state_old)[range(self.batch_size), action]
        q_next = self.dqn.forward(state_new)
        q_next[game_over] = 0.0
        q_target = reward + self.gamma*T.max(q_next, dim=1)[0]
        loss = self.dqn.loss(q_target, q).to(self.dqn.device)
        loss.backward()
        self.dqn.optimizer.step()
        self.epsilon = max(self.epsilon - self.eps_dec, self.eps_min)

    def save(self):
        self.dqn.save(self.episode_num, self.memory, self.epsilon, MODEL_PATH)

    def load(self, mode):
        self.episode_num, self.memory, self.epsilon = self.dqn.load(MODEL_PATH)
        if mode == 'train':
            print("Training Mode")
            self.dqn.train()
        elif mode == 'eval':
            print("Evaluation Mode")
            self.dqn.eval()

def train(dqn_agent):
    env = Environment()
    timer = time.time()
    while env.pygame_running():
        score = 0
        game_over = False
        state_old = env.get_state()
        while not game_over and score != AUTO_RESET:
            env.render()
            action = dqn_agent.get_action(state_old)
            reward, game_over = env.step(action)
            state_new = env.get_state()
            dqn_agent.remember(state_old, action, reward, state_new, game_over)
            dqn_agent.learn()
            state_old = state_new
            score += 1

        print(f"Episode: {dqn_agent.episode_num} Time: {time.time()-timer:.3f} Score: {score}")
        timer = time.time()
        
        env.reset()
        dqn_agent.episode_num += 1
        # save every 10 games
        if dqn_agent.episode_num % 10 == 0:
            dqn_agent.save()

if __name__ == '__main__':
    # Construct an argument parser
    all_args = argparse.ArgumentParser()
    all_args.add_argument("-train")
    all_args.add_argument("-reset")
    args = vars(all_args.parse_args())
    
    # initialize and load model
    dqn_agent = DQNAgent(gamma=.99, epsilon=1.0, lr=.003, batch_size=64)
    if args['train'] == '1':
        if args['reset'] != '1':  
            dqn_agent.load(mode='train')
            pass
        train(dqn_agent)
    else:
        dqn_agent.load(mode='eval')
        Environment().run(dqn_agent)
    
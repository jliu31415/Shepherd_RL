import numpy as np
from collections import deque
import random
from environment import Environment
from parameters import *
import torch as T
import time
import argparse

if CNN_NETWORK:
    from dqn_cnn import DQN
else:
    from dqn_linear import DQN

MODEL_PATH = 'model.pth'
# output/render while training
OUTPUT = True
RENDER = True

class DQNAgent:
    def __init__(self, gamma=.99, lr=.003, batch_size=64, max_mem_size=100000, 
                eps_start=0.9, eps_end=0.01, eps_decay=200):
        self.episode_num = 0
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.mem_size = max_mem_size
        self.memory = deque(maxlen=self.mem_size)
        if CNN_NETWORK:
            self.input_size = (FIELD_LENGTH, FIELD_LENGTH)
        else:
            self.input_size = 2*MAX_NUM_AGENTS+4
        self.output_size = 4
        self.batch_size = batch_size
        self.dqn = DQN(lr, self.input_size, self.output_size)
        self.target_dqn = DQN(lr, self.input_size, self.output_size)
        self.target_dqn.eval()

    def remember(self, state_old, action, reward, state_new, game_over):
        self.memory.append((state_old, action, reward, state_new, game_over))

    def get_action(self, state):
        # exploration vs exploitation
        eps_threshold = self.eps_end+(self.eps_start-self.eps_end)*\
            np.exp(-1.*self.episode_num/self.eps_decay)
        if random.random() < eps_threshold:
            action = random.randint(0, self.output_size-1)
        else:
            state_tensor = T.tensor(np.array(state), dtype=T.float).to(self.dqn.device)
            # add batch dimension
            state_tensor = T.unsqueeze(state_tensor, 0)
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
        
        q = self.dqn.forward(state_old)[range(self.batch_size), action]
        # expected values of actions computed based on the "older" target_dqn
        q_next = self.target_dqn.forward(state_new)
        q_next[game_over] = 0.0
        q_target = reward + self.gamma*T.max(q_next, dim=1)[0]
        loss = self.dqn.loss(q_target, q).to(self.dqn.device)
        
        # gradient descent
        self.dqn.optimizer.zero_grad()
        loss.backward()
        self.dqn.optimizer.step()

    def save(self):
        self.dqn.save(self.episode_num, MODEL_PATH)

    def load(self, mode):
        self.episode_num = self.dqn.load(MODEL_PATH)
        self.update_target_dqn()
        if mode == 'train':
            print("Training Mode")
            self.dqn.train()
        elif mode == 'eval':
            print("Evaluation Mode")
            self.dqn.eval()

    def update_target_dqn(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())

def train(dqn_agent):
    env = Environment(RENDER)
    timer = time.time()
    reward_memory = []
    num_wins = 0
    while not RENDER or env.pygame_running():
        score = 0
        episode_reward = 0
        game_over = False
        state_old = env.get_state()
        while not game_over and score != FRAME_RESET:
            if RENDER:
                env.render()
            action = dqn_agent.get_action(state_old)
            reward, game_over = env.step(action)
            state_new = env.get_state()
            dqn_agent.remember(state_old, action, reward, state_new, game_over)
            dqn_agent.learn()
            state_old = state_new
            score += 1
            episode_reward += reward

        reward_memory.append(episode_reward)
        indicator = "L "
        if score < FRAME_RESET:
            indicator = "W "
            num_wins += 1
        if OUTPUT:
            print(indicator + f"Episode {dqn_agent.episode_num}: time={time.time()-timer:.2f}s, " \
                    + f"score={score}, reward={episode_reward: .2f}")
        timer = time.time()
        
        env.reset()
        dqn_agent.episode_num += 1

        # update target network and save every x games
        if dqn_agent.episode_num % SAVE_TARGET == 0:
            dqn_agent.update_target_dqn()
            dqn_agent.save()
            print(f"Network saved on episode {dqn_agent.episode_num}, " \
                + f"avg reward={np.average(reward_memory):.2f}, " \
                + f"wins={num_wins}")
            reward_memory = []
            num_wins = 0

        if num_wins >= SAVE_TARGET*.8:
            break

if __name__ == '__main__':
    # Construct an argument parser
    all_args = argparse.ArgumentParser()
    all_args.add_argument("-train")
    all_args.add_argument("-reset")
    args = vars(all_args.parse_args())
    
    # initialize and load model
    dqn_agent = DQNAgent()
    if args['train'] == '1':
        if args['reset'] != '1':  
            dqn_agent.load(mode='train')
        train(dqn_agent)
    else:
        dqn_agent.load(mode='eval')
        Environment(True).run(dqn_agent)
    
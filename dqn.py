import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from threading import Thread

class DQN(nn.Module):
    def __init__(self, lr, input_size, output_size, hidden_size=256):
        super(DQN, self).__init__()
        self.lr = lr
        # two hidden layers
        self.input = nn.Linear(input_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x

    def save(self, episode_num, memory, epsilon, file_name):
        # create thread to prevent keyboard interrupt
        a = Thread(target=self.save_thread, args=(episode_num, memory, epsilon, file_name))
        a.start()
        a.join()

    def save_thread(self, episode_num, memory, epsilon, file_name):
        # save model
        folder_path = './model'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_name = os.path.join(folder_path, file_name)
        T.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode': episode_num,
            'memory': memory,
            'epsilon': epsilon
        }, file_name)

    # load model, and return episode number
    def load(self, file_name):
        checkpoint = T.load('./model/' + file_name)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['episode'], checkpoint['memory'], checkpoint['epsilon']

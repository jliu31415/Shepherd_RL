import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from threading import Thread

class DQN(nn.Module):
    def __init__(self, lr, input_size, output_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        # number of linear connections depends on conv2d layers (and therefore the input image size)
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(input_size[0]))
        convh = conv2d_size_out(conv2d_size_out(input_size[1]))
        linear_input_size = convw*convh*32
        self.linear1 = nn.Linear(linear_input_size, 128)
        self.linear2 = nn.Linear(128, 128)
        self.head = nn.Linear(128, output_size)

        self.lr = lr
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.head(x)

    def save(self, episode_num, file_name):
        # create thread to prevent keyboard interrupt
        a = Thread(target=self.save_thread, args=(episode_num, file_name))
        a.start()
        a.join()

    def save_thread(self, episode_num, file_name):
        # save model
        folder_path = './model'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_name = os.path.join(folder_path, file_name)
        T.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_num': episode_num
        }, file_name)

    # load model and return episode number
    def load(self, file_name):
        checkpoint = T.load('./model/' + file_name, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['episode_num']

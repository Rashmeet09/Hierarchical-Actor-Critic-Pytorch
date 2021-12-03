'''
Author: Rashmeet Kaur Nayyar
Hindsight Experience Replay (HER)
'''

from collections import deque
import torch
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Transitions sampled from the environment according to the exploration policy are stored in the Replay Buffer.
class ReplayBuffer():
    def __init__(self):
        self.max_size = 100000
        self.buffer = deque(maxlen=self.max_size)

    def add_experience(self, state, action, reward, next_state, subgoal, discount, done):
        # add the transition
        self.buffer.append((state, action, reward, next_state, subgoal, discount, done))

    def sample_experience(self, batch_size):
        # If the buffer is full, oldest samples can be discarded.
        if len(self.buffer) > self.max_size:
            del self.buffer[0:int(len(self.buffer)/5)]
        
        # sample transitions of size batch_size
        sampled_indices = np.random.randint(0, len(self.buffer), batch_size)
        batch = []
        for i in sampled_indices:
            batch.append(self.buffer[i])
        state, action, reward, next_state, subgoal, discount, done = map(np.stack, zip(*batch))
        
        # convert to tensors
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).reshape((batch_size,1)).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        subgoal = torch.FloatTensor(subgoal).to(device)
        discount = torch.FloatTensor(discount).reshape((batch_size,1)).to(device)
        done = torch.FloatTensor(done).reshape((batch_size,1)).to(device)
        return state, action, reward, next_state, subgoal, discount, done
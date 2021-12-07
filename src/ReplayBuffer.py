'''
Author: Rashmeet Kaur Nayyar
Hindsight Experience Replay (HER)
'''

import torch
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Transitions sampled from the environment according to the exploration policy are stored in the Replay Buffer.
class ReplayBuffer:
    def __init__(self, max_size=5e5):
        self.max_size = max_size
        self.buffer = []
        self.size = 0
    
    # adding experience in the form of tuple (state, action, reward, next_state, goal, discount, done)
    def add(self, transition): 
        self.buffer.append(transition)
        self.size +=1
    
    # sample experience in the form of state, action, reward, next_state, goal, discount, done
    def sample(self, batch_size):
        # If the buffer is full, oldest samples can be discarded.
        if self.size > self.max_size:
            del self.buffer[0:int(self.size/5)]
            self.size = len(self.buffer)

        # sample transitions of size batch_size
        sampled_indices = np.random.randint(0, len(self.buffer), size=batch_size)
        states, actions, rewards, next_states, goals, discounts, dones = [], [], [], [], [], [], []   
        for i in sampled_indices:
            states.append(np.array(self.buffer[i][0], copy=False))
            actions.append(np.array(self.buffer[i][1], copy=False))
            rewards.append(np.array(self.buffer[i][2], copy=False))
            next_states.append(np.array(self.buffer[i][3], copy=False))
            goals.append(np.array(self.buffer[i][4], copy=False))
            discounts.append(np.array(self.buffer[i][5], copy=False))
            dones.append(np.array(self.buffer[i][6], copy=False)) 

        # convert into tensors
        state = torch.FloatTensor(np.array(states)).to(device)
        action = torch.FloatTensor(np.array(actions)).to(device)
        reward = torch.FloatTensor(np.array(rewards)).reshape((batch_size,1)).to(device)
        next_state = torch.FloatTensor(np.array(next_states)).to(device)
        goal = torch.FloatTensor(np.array(goals)).to(device)
        discount = torch.FloatTensor(np.array(discounts)).reshape((batch_size,1)).to(device)
        done = torch.FloatTensor(np.array(dones)).reshape((batch_size,1)).to(device)
        return state, action, reward, next_state, goal, discount, done
    

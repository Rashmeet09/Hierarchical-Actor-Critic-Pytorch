'''
Author: Rashmeet Kaur Nayyar
Hierarchical Actor Critic (HAC)
Uses DDPG (Lilicrap et. al. 2015),
Universal Value Function Approximators (UVFA) (Schaul et al. 2015), and
Hindsight Experience Replay (HER) (Andrychowicz et al. 2017).
'''

import numpy as np
import torch
from src.DDPG import DDPG
from src.ReplayBuffer import ReplayBuffer
from matplotlib import animation
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HAC:
    def __init__(self, num_levels, max_horizon, state_dim, action_dim, render, subgoal_threshold, 
                 action_bounds, action_offset, state_bounds, state_offset, learning_rate):
        # initialize parameters
        self.goals = [None] * num_levels
        self.reward = 0
        self.timestep = 0

        # set parameters
        self.num_levels = num_levels
        self.max_horizon = max_horizon
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.subgoal_threshold = subgoal_threshold
        self.render = render

        # add layers of the hierarchy in bottom to top fashion
        # initialize experience buffer replay for each level
        self.HAC = [DDPG(state_dim, action_dim, action_bounds, action_offset, learning_rate, max_horizon)]
        self.replay_buffer = [ReplayBuffer()]
        for _ in range(num_levels-1):
            self.HAC.append(DDPG(state_dim, state_dim, state_bounds, state_offset, learning_rate, max_horizon))
            self.replay_buffer.append(ReplayBuffer())
        
    def set_parameters(self, lamda, gamma, action_clip_low, action_clip_high, 
                       state_clip_low, state_clip_high, exploration_action_noise, exploration_state_noise):     
        self.lamda = lamda
        self.gamma = gamma
        self.action_clip_low = action_clip_low
        self.action_clip_high = action_clip_high
        self.state_clip_low = state_clip_low
        self.state_clip_high = state_clip_high
        self.exploration_action_noise = exploration_action_noise
        self.exploration_state_noise = exploration_state_noise
    
    # Recursively calls itself to train each level of HAC
    def execute_HAC(self, env, i_level, state, goal, is_subgoal_test):
        # state_i and goal_i are current state and goal for level_i
        HER_storage = []
        self.goals[i_level] = goal
        next_state = None
        done = None
        for _ in range(self.max_horizon):
            # if this is a subgoal test, then next/lower level goal has to be a subgoal test
            is_next_subgoal_test = is_subgoal_test
            
            # To propose a subgoal action_i, sample action from the policy
            action = self.HAC[i_level].get_action_from_policy(state, goal)
            # If not bottom-most level
            if i_level > 0:
                # if not subgoal testing, take random action or add gaussian noise
                if not is_subgoal_test:
                    # 20% actions are sampled uniformly at random from the level's action space
                    # 80% actions are the sum of actions sampled from the levels's policy and gaussian noise
                    if np.random.random_sample() > 0.2:
                      action = action + np.random.normal(0, self.exploration_state_noise)
                      action = action.clip(self.state_clip_low, self.state_clip_high)
                    else:
                      action = np.random.uniform(self.state_clip_low, self.state_clip_high)
                
                # If layer_i is not bottom layer, then subgoal action_i is proposed for lower level to achieve
                # If layer_i is bottom layer, primitive action_i is executed
                # determine whether to test subgoal action_i
                if np.random.random_sample() < self.lamda:
                    is_next_subgoal_test = True
                
                # train (lower) level i-1 using subgoal action_i
                next_state, done = self.execute_HAC(env, i_level-1, state, action, is_next_subgoal_test)
                
                # when the level_i is not bottom layer and proposed subgoal action_i is not achieved by level i-1
                if is_next_subgoal_test and not self.is_goal(action, next_state, self.subgoal_threshold):
                    # subgoal testing transition: tests whether a proposed subgoal can be achieved by the lower level
                    # if subgoal action_i is not achieved by level i-1, level i is penalized with low penalty = - max_horizon  
                    self.replay_buffer[i_level].add((state, action, -self.max_horizon, next_state, goal, 0.0, float(done)))
                
                # hindsight action transition (replace the proposed action_i with the subgoal state achieved in hindsight)
                action = next_state
                
            # Bottom-most level
            else:
                if not is_subgoal_test:
                    if np.random.random_sample() > 0.2:
                      action = action + np.random.normal(0, self.exploration_action_noise)
                      action = action.clip(self.action_clip_low, self.action_clip_high)
                    else:
                      action = np.random.uniform(self.action_clip_low, self.action_clip_high)
                
                # execute primitive action
                next_state, reward, done, _ = env.step(action)
                if self.render:
                    if self.num_levels == 1:
                        env.render()
                    if self.num_levels == 2:
                        env.unwrapped.render_subgoals(self.goals[0], self.goals[1])   
                self.reward += reward
                self.timestep +=1
            
            # hindsight action transition
            goal_achieved = self.is_goal(next_state, goal, self.subgoal_threshold)
            if goal_achieved:
                self.replay_buffer[i_level].add((state, action, 0.0, next_state, goal, 0.0, float(done)))
            else:
                self.replay_buffer[i_level].add((state, action, -1.0, next_state, goal, self.gamma, float(done)))
                
            # hindsight goal transition
            HER_storage.append([state, action, -1.0, next_state, None, self.gamma, float(done)])
            
            state = next_state    
            if goal_achieved or done:
                break
        
        # Update the TBD component for the last HER transition
        # Perform HER using HER_storage_i transitions   
        HER_storage[-1][2] = 0.0
        HER_storage[-1][5] = 0.0
        for i in range(len(HER_storage)):
            HER_storage[i][4] = next_state
            self.replay_buffer[i_level].add(tuple(HER_storage[i]))
        
        return next_state, done
    
    def update_all_actor_critic_networks(self, n_iter, batch_size):
        for i in range(self.num_levels):
            self.HAC[i].update_actor_critic(self.replay_buffer[i], n_iter, batch_size)
            
    def is_goal(self, state, goal, subgoal_threshold):
        for i in range(self.state_dim):
            if abs(goal[i]-state[i]) > subgoal_threshold[i]:
                return False
        return True
    
    def save(self, directory, name):
        for i in range(self.num_levels):
            self.HAC[i].save(directory, name+'_level_{}'.format(i))
    
    def load(self, directory, name):
        for i in range(self.num_levels):
            self.HAC[i].load(directory, name+'_level_{}'.format(i))
    
    def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
        # can change frame size here
        plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
        anim.save(path + filename, writer='imagemagick', fps=60)
        
        
        
        

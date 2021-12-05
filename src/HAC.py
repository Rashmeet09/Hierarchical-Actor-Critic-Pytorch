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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HierarchicalActorCritic():

    def __init__(self, num_levels, max_horizon, state_dim, action_dim, subgoal_testing_rate, subgoal_threshold, render, discount, learning_rate, env_bounds):
        # initialize parameters
        self.HAC = list()                   # Hierarchical actor critic
        self.replay_buffers = list()        # Hindsight experience replay buffers
        self.reward = 0
        self.timesteps = 0
        self.subgoals = [None] * num_levels
        self.goal_reached = False
        self.test_subgoal = False

        # set parameters
        self.num_levels = num_levels                        # number of levels in the hierarchy
        self.max_horizon = max_horizon                      # maximum subgoal horizon
        self.state_dim = state_dim                          # state dimensionality
        self.action_dim = action_dim                        # action dimensionality
        self.subgoal_testing_rate = subgoal_testing_rate    # subgoal testing rate
        self.subgoal_threshold = subgoal_threshold          # subgoal achievement threshold
        self.render = render
        self.discount = discount
        self.learning_rate = learning_rate
        self.env_bounds = env_bounds      

        # add layers of the hierarchy in bottom to top fashion
        # initialize experience buffer replay for each level
        self.HAC.append(DDPG(self.state_dim, self.action_dim, self.env_bounds["action_max_bound"], self.env_bounds["action_offset"], self.max_horizon, self.learning_rate))
        self.replay_buffers.append(ReplayBuffer())
        for i in range(1,num_levels):
            self.HAC.append(DDPG(self.state_dim, self.state_dim, self.env_bounds["state_max_bound"], self.env_bounds["state_offset"], self.max_horizon, self.learning_rate))
            self.replay_buffers.append(ReplayBuffer())

    def execute(self, env, state, goal_state, n_iterations, batch_size):
        next_state, done = self.train_level(env, self.num_levels-1, state, goal_state, self.test_subgoal, batch_size)
        if self.is_goal(next_state, goal_state):
            self.goal_reached = True
        self.update_all_actor_critic_networks(n_iterations, batch_size)

    def train_level(self, env, level_i, state_i, goal_i, test_subgoal, batch_size):
        # state_i and goal_i are current state and goal for level_i
        self.subgoals[level_i] = goal_i
        HER_storage = list()
        h = 0
        next_state_i = None
        done = None
        while h <= self.max_horizon:

            # To propose a subgoal action_i, sample action from the policy
            action_i = self.HAC[level_i].get_action_from_policy(state_i, goal_i)
            # If subgoal testing flag is True, the lower level policy is followed exactly, else noise is added for exploration
            if test_subgoal == False:
                if level_i > 0:
                    # 20% actions are sampled uniformly at random from the level's action space
                    # 80% actions are the sum of actions sampled from the levels's policy and gaussian noise
                    if np.random.random() <= 0.2:
                        action_i = np.random.uniform(self.env_bounds["state_min_array"], self.env_bounds["state_max_array"])
                    else:
                        action_i += np.random.normal(0, self.env_bounds["state_exploration_std"])
                        action_i = action_i.clip(self.env_bounds["state_min_array"], self.env_bounds["state_max_array"])
                else:
                    if np.random.random() <= 0.2:
                        action_i = np.random.uniform(self.env_bounds["action_min_array"], self.env_bounds["action_max_array"])
                    else:
                        action_i += np.random.normal(0, self.env_bounds["action_exploration_std"])
                        action_i = action_i.clip(self.env_bounds["action_min_array"], self.env_bounds["action_max_array"])

            # If layer_i is not bottom layer, then subgoal action_i is proposed for lower level to achieve
            # If layer_i is bottom layer, primitive action_i is executed
            if level_i > 0:
                # determine whether to test subgoal action_i
                if np.random.random() < self.subgoal_testing_rate:
                    test_subgoal = True
                # train (lower) level i-1 using subgoal action_i
                next_state_i, done = self.train_level(env, level_i-1, state_i, action_i, test_subgoal, batch_size)
            else:
                next_state_i, reward, done, _ = env.step(action_i)
                self.timesteps += 1
                self.reward += reward
                if self.render:      
                    if self.num_levels == 2:
                        env.unwrapped.render_subgoals(self.subgoals[0], self.subgoals[1])
                    if self.num_levels == 3:
                        env.unwrapped.render_subgoals(self.subgoals[0], self.subgoals[1], self.subgoals[2])

            # when the level_i is not bottom layer and proposed subgoal action_i is not achieved by level i-1
            if level_i > 0 and not self.is_goal(next_state_i, action_i):
                # subgoal testing transition: tests whether a proposed subgoal can be achieved by the lower level
                # if subgoal action_i is not achieved by level i-1, level i is penalized with low penalty = - max_horizon
                if test_subgoal:
                    self.replay_buffers[level_i].add_experience(state_i, action_i, - self.max_horizon, next_state_i, goal_i, 0.0, float(done))
                # hindsight action transition (replace the proposed action_i with the subgoal state achieved in hindsight)
                action_i = next_state_i
            
            # hindsight action transition
            is_goal_achieved = self.is_goal(next_state_i, goal_i)
            if not is_goal_achieved:
                self.replay_buffers[level_i].add_experience(state_i, action_i, -1.0, next_state_i, goal_i, self.discount, float(done))
            else:
                self.replay_buffers[level_i].add_experience(state_i, action_i, 0.0, next_state_i, goal_i, 0.0, float(done))

            # hindsight goal transition
            HER_storage.append([state_i, action_i, -1.0, next_state_i, None, self.discount, float(done)])

            state_i = next_state_i
            h += 1
            if is_goal_achieved or done:
                break
           
        # Update the TBD component for the last HER transition
        # Perform HER using HER_storage_i transitions   
        for i in range(len(HER_storage)):
            if i == len(HER_storage)-1:
                HER_storage[i][2] == 0.0
                HER_storage[i][5] == 0.0
            HER_storage[i][4] = next_state_i
            self.replay_buffers[level_i].add_experience(HER_storage[i][0], HER_storage[i][1], HER_storage[i][2], HER_storage[i][3], HER_storage[i][4], HER_storage[i][5], HER_storage[i][6])
        
        return next_state_i, done

    def update_all_actor_critic_networks(self, n_iterations, batch_size):
        for level_id in range(self.num_levels):
            self.HAC[level_id].update_actor_critic(self.replay_buffers[level_id], n_iterations, batch_size)

    def is_goal(self, next_state, goal):
        is_goal_reached = True
        for i in range(self.state_dim):
            if abs(goal[i] - next_state[i]) > self.subgoal_threshold[i]:
                is_goal_reached = False
                break
        return is_goal_reached
    
    def save_model(self, model_directory):
        for level_id in range(self.num_levels):
            torch.save(self.HAC[level_id].actor.state_dict(), '{}/actor_level_{}.pth'.format(model_directory, level_id))
            torch.save(self.HAC[level_id].critic.state_dict(), '{}/critic_level_{}.pth'.format(model_directory, level_id))

    def load_model(self, model_directory):
        for level_id in range(self.num_levels):
            self.HAC[level_id].actor.load_state_dict(torch.load('{}/actor_level_{}.pth'.format(model_directory, level_id)))
            self.HAC[level_id].critic.load_state_dict(torch.load('{}/critic_level_{}.pth'.format(model_directory, level_id)))
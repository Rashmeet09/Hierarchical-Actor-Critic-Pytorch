'''
Author: Rashmeet Kaur Nayyar
Hierarchical Actor Critic (HAC)
Uses DDPG (Lilicrap et. al. 2015),
Universal Value Function Approximators (UVFA) (Schaul et al. 2015), and
Hindsight Experience Replay (HER) (Andrychowicz et al. 2017).
'''

import os
import numpy as np
import torch
import gym
import time
import gym_custom
from src.HAC import HierarchicalActorCritic
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# np.random.seed(int(time.time()))
# np.random.seed(10)

if __name__=="__main__":
    # training configuration
    num_episodes_train = 1000
    num_episodes_test = 10
    interval_episode = 2
    n_iterations = 100
    batch_size = 32
    discount = 0.9
    learning_rate = 0.001
    render = True
    train = True
    test = True

    # HAC Specific
    num_levels = 2   # or 3
    max_horizon = 20     
    subgoal_testing_rate = 0.3
    test_subgoal = False

    # Environment Specific
    env_name = 'MountainCarContinuous-v1'
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]  # Joint positions and velocities
    action_dim = env.action_space.shape[0]      # Joint torques
    goal_state = np.array([0.46, 0.02])         # Array of joint position and velocity
    subgoal_threshold = np.array([0.01, 0.02])      
    env_bounds = dict()
    # action bounds and offset
    env_bounds["action_max_bound"] = env.action_space.high[0]
    env_bounds["action_min_array"] = np.array([-1.0]) # or np.array([env.action_space.low[0]])
    env_bounds["action_max_array"] = np.array([1.0])  # or np.array([env.action_space.high[0]])  
    env_bounds["action_offset"] = torch.FloatTensor(np.array([0.0]).reshape(1, -1)).to(device)
    # state bounds and offset
    env_bounds["state_max_bound"] = torch.FloatTensor(np.array([0.6, 0.07]).reshape(1, -1)).to(device)
    env_bounds["state_min_array"] = np.array([-1.2, -0.07])
    env_bounds["state_max_array"] = np.array([0.6, 0.07])  
    env_bounds["state_offset"] = torch.FloatTensor(np.array([-0.3, 0.0]).reshape(1, -1)).to(device)
    # std of gaussian noise for exploration policy
    env_bounds["action_exploration_std"] = np.array([0.1])        
    env_bounds["state_exploration_std"] = np.array([0.02, 0.01]) 

    model_directory = os.getcwd()+"/model/"+env_name
    train_log_file = open(model_directory+"/train_log.txt", "w+")
    test_log_file = open(model_directory+"/test_log.txt", "w+")

    # Initialize HAC, train it, and print rewards for each episode
    hac_agent = HierarchicalActorCritic(num_levels, max_horizon, state_dim, action_dim, subgoal_testing_rate, subgoal_threshold, test_subgoal, render, discount, learning_rate, env_bounds)   

    if train:
        start_time = time.time()
        for episode in range(1,num_episodes_train+1):
            hac_agent.reward = 0
            hac_agent.timesteps = 0
            state = env.reset()
            reward, goal_reached = hac_agent.train(env, state, goal_state, n_iterations, batch_size)
            if episode % interval_episode == 0:
                hac_agent.save_model(model_directory)
            if goal_reached:
                print("The agent solved the problem!")
            time_taken = round(time.time()-start_time,2)
            train_log_file.write("Episode: {}, Reward: {}, Time taken: {} s".format(episode, reward, time_taken))
            print("Episode: {}, Reward: {}, Time taken: {} s".format(episode, reward, time_taken))
        env.close()

    if test:
        hac_agent.load_model(model_directory)
        for episode in range(1,num_episodes_test+1):
            hac_agent.reward = 0
            hac_agent.timesteps = 0
            state = env.reset()
            reward, goal_reached = hac_agent.train(env, state, goal_state, n_iterations, batch_size)
            if goal_reached:
                print("The agent solved the problem!")
            time_taken = round(time.time()-start_time,2)
            test_log_file.write("Episode: {}, Reward: {}, Time taken: {} s".format(episode, reward, time_taken))
            print("Episode: {}, Reward: {}, Time taken: {} s".format(episode, reward, time_taken))
        env.close()

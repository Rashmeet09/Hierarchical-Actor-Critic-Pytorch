'''
Author: Rashmeet Kaur Nayyar
Hierarchical Actor Critic (HAC)
Uses DDPG (Lilicrap et. al. 2015),
Universal Value Function Approximators (UVFA) (Schaul et al. 2015), and
Hindsight Experience Replay (HER) (Andrychowicz et al. 2017).
'''

import os
import sys
import numpy as np
import torch
import gym
import time
import matplotlib.pyplot as plt
import pickle
import gc
import gym_custom
from src.HAC import HierarchicalActorCritic
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def plot_success_rate(model_directory, env_name, plotname, pickle_file):
    SMALL_SIZE = 9
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=20)             # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)
    plt.rc('font.sans-serif')

    with open(pickle_file, 'rb') as handle:
        episode_to_sucess_rate = pickle.load(handle)

    fig, ax1 = plt.subplots()
    l1 = ax1.plot(episode_to_sucess_rate.keys(), episode_to_sucess_rate.values(), color="tab:red", label="Number of Queries by AIA")[0]
    plt.xlabel("Training number of episodes")
    plt.ylabel("Avg success rate")

    plt.title(env_name)
    plt.tight_layout()
    plt.savefig(model_directory + "/" + plotname)

if __name__=="__main__":
    # Training configuration
    num_episodes_train = 100
    num_episodes_test = 5
    interval_episode = 10
    n_iterations = 100
    batch_size = 100
    discount = 0.95
    learning_rate = 0.001
    random_seed = 10         # or int(time.time())
    render = False
    train = True
    test = True

    # HAC parameters
    num_levels = 2  # or 3
    max_horizon = 20     
    subgoal_testing_rate = 0.3

    # Environment parameters
    env_name = sys.argv[1]
    # env_name = 'MountainCarContinuous-v1'
    # env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    env_bounds = dict()
    if env_name == 'MountainCarContinuous-v1':
        state_dim = env.observation_space.shape[0]       # Joint positions and velocities
        action_dim = env.action_space.shape[0]           # Joint torques
        goal_state = np.array([0.46, 0.04])              # Array of joint position and velocity
        subgoal_threshold = np.array([0.01, 0.02])      
        # action bounds and offset used for normalization
        env_bounds["action_max_bound"] = env.action_space.high[0]
        env_bounds["action_min_array"] = np.array([env.action_space.low[0]])
        env_bounds["action_max_array"] = np.array([env.action_space.high[0]])  
        env_bounds["action_offset"] = torch.FloatTensor(np.array([0.0]).reshape(1, -1)).to(device)
        # state bounds and offset used for normalization (values come from the original Open AI Gym's MountainCarContinuous environment)
        env_bounds["state_max_bound"] = torch.FloatTensor(np.array([0.6, 0.07]).reshape(1, -1)).to(device)
        env_bounds["state_min_array"] = np.array([-1.2, -0.07])
        env_bounds["state_max_array"] = np.array([0.6, 0.07])  
        env_bounds["state_offset"] = torch.FloatTensor(np.array([-0.3, 0.0]).reshape(1, -1)).to(device)
        # std of gaussian noise for exploration policy
        env_bounds["action_exploration_std"] = np.array([0.1])        
        env_bounds["state_exploration_std"] = np.array([0.02, 0.01]) 
    elif env_name == 'Pendulum-v1':
        state_dim = 2 # or env.observation_space.shape[0]  # Joint positions and velocities
        action_dim = env.action_space.shape[0]             # Joint torques
        goal_state = np.array([0.0, 0.0])                  # Array of joint position and velocity
        subgoal_threshold = np.array([np.deg2rad(10), 0.05])      
        # action bounds and offset
        env_bounds["action_max_bound"] = env.action_space.high[0]
        env_bounds["action_min_array"] = np.array([env.action_space.low[0]])
        env_bounds["action_max_array"] = np.array([env.action_space.high[0]])  
        env_bounds["action_offset"] = torch.FloatTensor(np.array([0.0]).reshape(1, -1)).to(device)
        # state bounds and offset
        env_bounds["state_max_bound"] = torch.FloatTensor(np.array([np.pi, 8.0]).reshape(1, -1)).to(device)
        env_bounds["state_min_array"] = np.array([-np.pi, -8.0])
        env_bounds["state_max_array"] = np.array([np.pi, 8.0])  
        env_bounds["state_offset"] = torch.FloatTensor(np.array([0.0, 0.0]).reshape(1, -1)).to(device)
        # std of gaussian noise for exploration policy
        env_bounds["action_exploration_std"] = np.array([0.1])        
        env_bounds["state_exploration_std"] = np.array([np.deg2rad(10), 0.4]) 

    # fix random seed   
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    env.seed(random_seed)

    # File paths
    model_directory = os.getcwd()+"/model/"+env_name+"/num_levels_"+str(num_levels)
    train_log_file = model_directory+"/train_log.txt"
    test_log_file = model_directory+"/test_log.txt"
    with open(train_log_file, "w+") as f:
        f.write("Episode, Reward, Timesteps, Time taken (s)\n")
    with open(test_log_file, "w+") as f:
        f.write("Episode, Reward, Timesteps, Time taken (s)\n")
    train_success_rate_pickle = model_directory+"/"+"train_success_rate_pickle.pkl"
    test_success_rate_pickle = model_directory+"/"+"test_success_rate_pickle.pkl"

    # Initialize HAC agent
    hac_agent = HierarchicalActorCritic(num_levels, max_horizon, state_dim, action_dim, subgoal_testing_rate, subgoal_threshold, render, discount, learning_rate, env_bounds)   

    # Train HAC agent
    gc.collect()
    torch.cuda.empty_cache()
    if train:
        epsiode_to_success_rate = dict()
        successful_episodes = 0
        start_time = time.time()

        for episode in range(1,num_episodes_train+1):
            # reset values
            hac_agent.reward = 0
            hac_agent.timesteps = 0
            hac_agent.goal_reached = False
            state = env.reset()

            # execute HAC agent
            hac_agent.execute(env, state, goal_state, n_iterations, batch_size)
            
            # check if the goal reached
            if hac_agent.goal_reached:
                print("The agent solved the problem!")
                successful_episodes += 1

            # save the model in intervals
            if episode % interval_episode == 0 or hac_agent.goal_reached:
                hac_agent.save_model(model_directory)
            
            # log reward, timesteps, and time taken for each episode
            time_taken = round(time.time()-start_time,2)
            with open(train_log_file, "a+") as f:
                f.write("Episode: {}, Reward: {}, Timesteps: {}, Time taken: {} s\n".format(episode, round(hac_agent.reward,2), hac_agent.timesteps, time_taken))
            print("Episode: {}, Reward: {}, Timesteps: {}, Time taken: {} s".format(episode, round(hac_agent.reward,2), hac_agent.timesteps, time_taken))
            
            # log epsiode to success_rate
            epsiode_to_success_rate[episode] = successful_episodes
            with open(train_success_rate_pickle, 'wb') as handle:
                pickle.dump(epsiode_to_success_rate, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        print("Success rate:",successful_episodes*100/num_episodes_train)
        env.close()
        plot_success_rate(model_directory, env_name, "_train.png", train_success_rate_pickle)

    # Test HAC agent
    if test:
        epsiode_to_success_rate = dict()
        successful_episodes = 0
        start_time = time.time()

        # load the trained HAC agent
        hac_agent.load_model(model_directory)
        for episode in range(1,num_episodes_test+1):
            # reset values
            hac_agent.reward = 0
            hac_agent.timesteps = 0
            hac_agent.goal_reached = False
            state = env.reset()

            # execute HAC agent
            hac_agent.execute(env, state, goal_state, n_iterations, batch_size)
            
            # check if the goal reached
            if hac_agent.goal_reached:
                print("The agent solved the problem!")
                successful_episodes += 1

            # log reward, timesteps, and time taken for each episode
            time_taken = round(time.time()-start_time,2)
            with open(test_log_file, "a+") as f:
                f.write("Episode: {}, Reward: {}, Timesteps: {}, Time taken: {} s\n".format(episode, round(hac_agent.reward,2), hac_agent.timesteps, time_taken))
            print("Episode: {}, Reward: {}, Timesteps: {}, Time taken: {} s".format(episode, round(hac_agent.reward,2), hac_agent.timesteps, time_taken))
            
            # log epsiode to success_rate
            epsiode_to_success_rate[episode] = successful_episodes
            with open(test_success_rate_pickle, 'wb') as handle:
                pickle.dump(epsiode_to_success_rate, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        print("Success rate:",successful_episodes*100/num_episodes_test)
        env.close()
        plot_success_rate(model_directory, env_name, "_test.png", test_success_rate_pickle)

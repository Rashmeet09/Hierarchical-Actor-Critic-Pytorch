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
import matplotlib.pyplot as plt
import pickle
import gym_custom
from src.HAC import HAC
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

if __name__ == '__main__':
    env_name = "MountainCarContinuous-v1" # or "Pendulum-v1"
    num_levels = 2                       # number of levels in hierarchy
    max_horizon = 20                     # maximum horizon to achieve subgoal
    subgoal_testing_rate = 0.3           # subgoal testing rate
                
    num_episodes_train = 1000     
    num_episodes_test = 10     
    interval_episode = 50           
    num_iterations = 100              
    batch_size = 100           
    learning_rate = 0.001
    discount = 0.95                
    random_seed = 0
    render = False
    train = True
    test = True
  
    env = gym.make(env_name)
    if env_name == "MountainCarContinuous-v1":   
        state_dim = env.observation_space.shape[0]   # Joint positions and velocities
        action_dim = env.action_space.shape[0]       # Joint torques 
        goal_state = np.array([0.48, 0.04])          # Array of joint position and velocity
        threshold = np.array([0.01, 0.02])           # sub-goal threshold value to test if sub-goal is achieved by lower level  
        # action bounds and offset used for normalization
        action_bounds = env.action_space.high[0]
        action_clip_low = np.array([-1.0 * action_bounds])
        action_clip_high = np.array([action_bounds]) 
        action_offset = torch.FloatTensor(np.array([0.0]).reshape(1, -1)).to(device)
        # state bounds and offset used for normalization (values come from the original Open AI Gym's MountainCarContinuous environment)
        state_bounds = torch.FloatTensor(np.array([0.9, 0.07]).reshape(1, -1)).to(device)
        state_clip_low = np.array([-1.2, -0.07])
        state_clip_high = np.array([0.6, 0.07])
        state_offset = torch.FloatTensor(np.array([-0.3, 0.0]).reshape(1, -1)).to(device)
        # std of gaussian noise for exploration policy
        exploration_action_noise = np.array([0.1])        
        exploration_state_noise = np.array([0.02, 0.01])   
    elif env_name == 'Pendulum-v1':
        state_dim = 2 # or env.observation_space.shape[0]  # Joint positions and velocities
        action_dim = env.action_space.shape[0]             # Joint torques
        goal_state = np.array([0.0, 0.0])                  # Array of joint position and velocity
        threshold = np.array([np.deg2rad(10), 0.05])      
        # action bounds and offset used for normalization
        action_bounds = env.action_space.high[0]
        action_clip_low = np.array([env.action_space.low[0]])
        action_clip_high = np.array([env.action_space.high[0]])  
        action_offset = torch.FloatTensor(np.array([0.0]).reshape(1, -1)).to(device)
        # state bounds and offset used for normalization (values come from the original Open AI Gym's Pendulum environment)
        state_bounds = torch.FloatTensor(np.array([np.pi, 8.0]).reshape(1, -1)).to(device)
        state_clip_low = np.array([-np.pi, -8.0])
        state_clip_high = np.array([np.pi, 8.0])  
        state_offset = torch.FloatTensor(np.array([0.0, 0.0]).reshape(1, -1)).to(device)
        # std of gaussian noise for exploration policy
        exploration_action_noise = np.array([0.1])        
        exploration_state_noise = np.array([np.deg2rad(10), 0.4]) 
    
    if random_seed:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    # File paths
    model_directory = os.getcwd()+"/model/{}/{}level".format(env_name, num_levels) 
    filename = "HAC_{}".format(env_name)
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    train_log_file = model_directory+"/train_log.txt"
    test_log_file = model_directory+"/test_log.txt"
    with open(train_log_file, "w+") as f:
        f.write("Episode, Reward, Timesteps, Time taken (s)\n")
    with open(test_log_file, "w+") as f:
        f.write("Episode, Reward, Timesteps, Time taken (s)\n")
    train_success_rate_pickle = model_directory+"/"+"train_success_rate_pickle.pkl"
    test_success_rate_pickle = model_directory+"/"+"test_success_rate_pickle.pkl"
    
    # Initialize HAC agent
    hac_agent = HAC(num_levels, max_horizon, state_dim, action_dim, render, threshold, 
                action_bounds, action_offset, state_bounds, state_offset, learning_rate)
    
    # Set parameters
    hac_agent.set_parameters(subgoal_testing_rate, discount, action_clip_low, action_clip_high, 
                       state_clip_low, state_clip_high, exploration_action_noise, exploration_state_noise)
    
    if train:
        # Train HAC agent
        epsiode_to_success_rate = dict()
        successful_episodes = 0
        start_time = time.time()

        for i_episode in range(1, num_episodes_train+1):
            # reset values
            hac_agent.reward = 0
            hac_agent.timestep = 0  
            state = env.reset()

            # execute HAC agent
            last_state, done = hac_agent.execute_HAC(env, num_levels-1, state, goal_state, False)
            
            # check if the goal reached
            if hac_agent.is_goal(last_state, goal_state, threshold):
                print("The agent solved the problem!")
                hac_agent.save(model_directory, filename + '_final')
                successful_episodes += 1
            
            # update all actor critic networks
            hac_agent.update_all_actor_critic_networks(num_iterations, batch_size)

            # save the model in intervals
            if i_episode % interval_episode == 0:
                hac_agent.save(model_directory, filename)
            
            # log reward, timesteps, and time taken for each episode
            time_taken = round(time.time()-start_time,2)
            with open(train_log_file, "a+") as f:
                f.write("Episode: {}, Reward: {}, Timesteps: {}, Time taken: {} s\n".format(i_episode, round(hac_agent.reward,2), hac_agent.timestep, time_taken))
            print("Episode: {}, Reward: {}, Timesteps: {}, Time taken: {} s".format(i_episode, round(hac_agent.reward,2), hac_agent.timestep, time_taken))
            
            # log epsiode to number of successful episodes
            epsiode_to_success_rate[i_episode] = successful_episodes
            with open(train_success_rate_pickle, 'wb') as handle:
                pickle.dump(epsiode_to_success_rate, handle, protocol=pickle.HIGHEST_PROTOCOL)        
    
        env.close()
        print("Success rate:",successful_episodes*100/num_episodes_train)
        # plot_success_rate(model_directory, env_name, "_test.png", test_success_rate_pickle)

    if test:
        # load agent
        hac_agent.load(model_directory, filename)

        # Test HAC agent
        epsiode_to_success_rate = dict()
        successful_episodes = 0
        start_time = time.time()

        for i_episode in range(1, num_episodes_test+1):
            # reset values
            hac_agent.reward = 0
            hac_agent.timestep = 0      
            state = env.reset()

            last_state, done = hac_agent.execute_HAC(env, num_levels-1, state, goal_state, True)
            
            # check if the goal reached
            if hac_agent.is_goal(last_state, goal_state, threshold):
                print("The agent solved the problem!")
                hac_agent.save(model_directory, filename + "final")
                successful_episodes += 1

            # log reward, timesteps, and time taken for each episode
            time_taken = round(time.time()-start_time,2)
            with open(test_log_file, "a+") as f:
                f.write("Episode: {}, Reward: {}, Timesteps: {}, Time taken: {} s\n".format(i_episode, round(hac_agent.reward,2), hac_agent.timestep, time_taken))
            print("Episode: {}, Reward: {}, Timesteps: {}, Time taken: {} s".format(i_episode, round(hac_agent.reward,2), hac_agent.timestep, time_taken))
            
            # log epsiode to number of successful episodes
            epsiode_to_success_rate[i_episode] = successful_episodes
            with open(test_success_rate_pickle, 'wb') as handle:
                pickle.dump(epsiode_to_success_rate, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        env.close()
        print("Success rate:",successful_episodes*100/num_episodes_test)
        # plot_success_rate(model_directory, env_name, "_test.png", test_success_rate_pickle)
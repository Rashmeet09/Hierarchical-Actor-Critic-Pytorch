import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') 
import pickle

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
        episode_to_success_rate = pickle.load(handle)
    # success_rate = episode_to_success_rate.values()
    success_rate = [list(episode_to_success_rate.values())[i]*100.0/list(episode_to_success_rate.keys())[i] for i in range(len(episode_to_success_rate))]

    fig, ax1 = plt.subplots()
    l1 = ax1.plot(episode_to_success_rate.keys(), success_rate, color="tab:red", label="Number of Queries by AIA")[0]
    plt.xlabel("Training number of episodes")
    plt.ylabel("Avg success rate")

    plt.title(env_name)
    plt.tight_layout()
    plt.savefig(model_directory + "/" + plotname)

def plot_for_all_levels(model_directory, level_to_pickle):
    SMALL_SIZE = 9
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=10)             # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)
    plt.rc('font.sans-serif')

    level_to_episode_to_success_rate = dict()
    for level, pickle_file in level_to_pickle.items():
        with open(pickle_file, 'rb') as handle:
            episode_to_num_success = pickle.load(handle)
        episode_to_success_rate = dict()
        for epi, n in episode_to_num_success.items():
            episode_to_success_rate[epi] = n*100.0/epi
        level_to_episode_to_success_rate[level] = episode_to_success_rate

    fig, ax1 = plt.subplots()
    l1 = ax1.plot(level_to_episode_to_success_rate[1].keys(), level_to_episode_to_success_rate[1].values(), color="tab:red", label="1 level")[0]
    l2 = ax1.plot(level_to_episode_to_success_rate[2].keys(), level_to_episode_to_success_rate[2].values(), color="tab:blue", label="2 levels")[0]
    l3 = ax1.plot(level_to_episode_to_success_rate[3].keys(), level_to_episode_to_success_rate[3].values(), color="tab:green", label="3 levels")[0]

    plt.xlabel("Training number of episodes")
    plt.ylabel("Average success rate")
    plt.legend()  

    plt.title(env_name)
    plt.tight_layout()
    plt.savefig(model_directory + "/comparision.png" )

if __name__=="__main__":
     # save trained models
    # env_name = "MountainCarContinuous-v1"
    env_name = "Pendulum-v1"
    
    # k_level = 3
    # model_directory = os.getcwd()+"/model/{}/{}level/".format(env_name, k_level) 
    # filename = "HAC_{}".format(env_name)
    # train_log_file = model_directory+"/train_log.txt"
    # test_log_file = model_directory+"/test_log.txt" 
    # train_success_rate_pickle = model_directory+"/"+"train_success_rate_pickle.pkl"
    # test_success_rate_pickle = model_directory+"/"+"test_success_rate_pickle.pkl"
    # plot_success_rate(model_directory, env_name, "_train.png", train_success_rate_pickle)

    level_to_pickle = dict()
    for k_level in [1,2,3]:
        model_directory = os.getcwd()+"/model/{}/{}level/".format(env_name, k_level) 
        train_success_rate_pickle = model_directory+"/"+"train_success_rate_pickle.pkl"
        level_to_pickle[k_level] =  train_success_rate_pickle
    plot_for_all_levels(os.getcwd()+"/model/{}/".format(env_name), level_to_pickle)
# Hierarchical-Actor-Critic-Pytorch

Hierarchical Actor Critic (HAC) helps agents learn tasks more quickly by enabling them to break problems down into short sequences of actions. It uses 
1. DDPG (Lilicrap et. al. 2016),
2. Universal Value Function Approximators (UVFA) (Schaul et al. 2015), and
3. Hindsight Experience Replay (HER) (Andrychowicz et al. 2017).

## Use the following commands to train and test
You can modify the training and testing configuration and the parameters of HAC and DDPG in the main.py file. The parameters have been explained below. You can find the result plots in the Results/ directory. The script install.sh has commands needed to run for installing prerequisite packages.

```
python3 main.py
```
1. env_name: "MountainCarContinuous-v1" or "Pendulum-v1"
2. num_levels: number of levels in hierarchy
3. max_horizon: maximum horizon to achieve subgoal
4. subgoal_testing_rate: subgoal testing rate     
5. num_episodes_train: maximum number of episodes to train
6. num_episodes_test: maximum number of episodes to test  
7. interval_episode: interval of episodes to save the models          
8. num_iterations: number of iterations            
9. batch_size: size of batch          
10. learning_rate: rate of learning
11. discount: discount rate to use for future rewards             
12. random_seed: random seed
13. render: flag for rendering/visualizing the environment
14. train: flag to train the HAC agent (saves the model)
15. test: flag to test the HAC agent (loads the saved model)

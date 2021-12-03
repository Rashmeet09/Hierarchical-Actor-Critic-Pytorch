# Hierarchical-Actor-Critic-Pytorch

Hierarchical Actor Critic (HAC) helps agents learn tasks more quickly by enabling them to break problems down into short sequences of actions. It uses 
1] DDPG (Lilicrap et. al. 2016),
2] Universal Value Function Approximators (UVFA) (Schaul et al. 2015), and
3] Hindsight Experience Replay (HER) (Andrychowicz et al. 2017).

Deep Deterministic Policy Gradient (DDPG) is an actor-critic, model-free, off-policy algorithm to learn a policy over continuous action domains. It was proposed by Lillicrap et. al. 2016 after the success of Deep Q Network (DQN) for discrete action domains in Mnih et. al. 2015. DDPG is based on Deep Deterministic Gradient (DPG) actor-critic algorithm proposed by Silver et. al. 2014.
Innovations of DQN:
 1. the network is trained off-policy with samples from a "replay buffer" to minimize correlations between samples; 
 2. the network is trained with a separate "target Q network" to give consistent targets during temporal difference backups.

References:
[Andrew Levy et. al. 2016](https://arxiv.org/pdf/1712.00948.pdf)
[Andrew Levy et. al. 2018](https://blogs.cuit.columbia.edu/zp2130/files/2019/02/Hierarchical-Actor-Critic.pdf)
[Lilicrap et. al. 2016](https://arxiv.org/pdf/1509.02971v6.pdf)
[Tensorflow github by Andrew](https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-)
[PyTorch github by Nikhil](https://github.com/nikhilbarhate99/Hierarchical-Actor-Critic-HAC-PyTorch/tree/07137e260b89a299e5a3025e11c33f3bcb5e7890)
[Blog by Andrew](http://bigai.cs.brown.edu/2019/09/03/hac.html)
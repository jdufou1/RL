# ES-RL/RL Algorithms

## Value-based methods
In this repo, you could find the implementation of Reinforcement Learning algorithms. I have started with the two most famous RL model-based algorithms : Policy and Value Iteration. Also, I have implemented model-free algorithms such as Q-Learning / Double Q-Learning (with two estimators to reduce overestimation), SARSA / Expected SARSA , TD($\lambda$) with eligibility traces, Actor-Critic and Monte-Carlo method. The experimentations are available in ``tests\tests_value_based.ipynb``.

<p align="center">
  <img src="https://github.com/jdufou1/RL/blob/main/img/test_taxiv3_value_based.png" height="250px"/>
</p>

## Double Q-Learning for Flappy Bird
I used the Double-Q-Learning to make a flappy bird agent. Therefore, I had to discretize the observation space : (x,y,speed)
where x : horizontal distance between the bird and the agent, y : the vertical distance and the vertical-speed of the bird.
So, I round the value to make these values tabular. The learning took me about 8/9 hours with CPU and the agent was capable to reach an average score of 45.

<p align="center">
  <img src="https://github.com/jdufou1/RL/blob/main/img/flappy_bird.gif" alt="animated" height="450px"/>
</p>

## DQN-family
Then, I work on the DQN to understand why this family works very well on the ATARI games. So, I read some papers and I have implemented three versions : vanilla DQN , Double/Dueling DQN,  Double DQN with Prioritized Experience Replay buffer and D3QN. The last one allows me to learn parameters for the flappy bird game. The experimentations are available in ``tests\tests_dqn_family_lunarlander.ipynb``.

<div style="display:flex;">
    <img src="https://github.com/jdufou1/RL/blob/main/img/test_lunarlander_dqn.png" alt="drawing" width="43%"/>
    <img src="https://github.com/jdufou1/RL/blob/main/img/lunarlander.gif" alt="drawing" width="45%"/>
</div>


## Gradient-based methods
All of these algorithms are value-based and the policy is directly derivative from the value-function (state or action).An other possibility is to work directly on the parameters of the policy $\pi_{\theta}$. Therefore, I have implemented gradient-based (Deep RL) algorithms such that Reinforce, A2C (actor-critic method : we learn value-function and policy iteratively), PPO and DDPG. I have used Pytorch to build my neural networks.

## Evolution Strategies methods for RL environments 

These methods are very differents of the RL-methods and work like black-box. Now, we will stop to make gradient operation or learn a value-function because we will apply a gaussian noise on the parameters and learn with it. So, it requires more sampling. Firslr, I have implemented basics algorithms to find the minimum of some loss function : Naive (1+1)-ES , ($\mu=$m + $\lambda=$n)-ES , Cross-Entropy Method (CEM) and so on... then I apply these methods on the RL environment : Walker2d-v2, Humanoid-v0 from MuJoCo. The experimentations are available in ``ES\EvolutionaryStrategy.ipynb``.

## Source

- https://lilianweng.github.io/posts/2018-04-08-policy-gradient/
- https://philippe-preux.github.io/Documents/digest-ar.pdf
- https://cs229.stanford.edu/proj2015/362_report.pdf
- https://arxiv.org/abs/1509.06461v3
- https://arxiv.org/abs/1509.02971

# ES-RL/RL Algorithms

## Value-based methods
In this repo, you could find the implementation of Reinforcement Learning algorithms. I have started with the two most famous RL model-based algorithms : Policy and Value Iteration. Also, I have implemented model-free algorithms such that Q-Learning / Double Q-Learning (with two estimators to reduce overestimation), SARSA , TD($\lambda$) with eligibility traces and Monte-Carlo method.


## Double Q-Learning for Flappy Bird
I used the Double-Q-Learning to make a flappy bird agent. Therefore, I have to discretize the observation space : (x,y,speed)
where x : horizontal distance between the bird and the agent, y : the vertical distance and the vertical-speed of the bird.
So, I round the value to make these tabular values. The learning took me about 8/9 hours and the agent was capable to reach a mean score of 80. The project is here.

## DQN-family
Then, I work on the DQN to understand why this family works very well on the ATARI games. So, I read some papers and I have implemented three versions : vanilla DQN , Double DQN, Double DQN with Prioritized Experience Replay buffer. The last one allows me to learns parameters for the flappy bird game from the RGB representation.

## Gradient-based methods
All of these algorithms are value-based and the policy is directly derivative from the value-function (state or action).An other possibility is to work directly on the parameters of the policy $\pi_{\theta}$. Therefore, I have implemented gradient-based (Deep RL) algorithms such that Reinforce, A2C (actor-critic method : we learn value-function and policy iteratively), PPO and DDPG. I have used Pytorch to build my neural networks.

## Evolution Strategies methods for RL environments 

These methods are very differents of the RL-methods and work like black-box. Now, we will stop to make gradient operation or learn a value-function because we will apply a gaussian noise on the parameters and learn with it. So, it requires more sampling. Firslr, I have implemented basics algorithms to find the minimum of some loss function : Naive (1+1)-ES , ($\mu=$m + $\lambda=$n)-ES , Cross-Entropy Method (CEM) and so on... then I apply this method on the RL environment : Walker2d-v2, Humanoid-v0 from MuJoCo. You could find my work on this notebook here.

## Source


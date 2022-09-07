# D3QN for Flappy Bird environment

In this repo, I have implemented a Dueling Double DQN (D3QN) in order to learn to play flappy bird.

The states are composed by the vertical and the horizontal distance between the bird and the nearest pipe, the y-speed of the bird $\in {-9,-8,...,8,9}$ and the bird's rotation. So 4 dimensions are used.

Here are the hyper-parameters and the reward function that I used :
- learning rate : 2e-3
- batch size : 128
- discount factor : 0.99
- test frequency : 10
- epsilon min : 0.02
- epsilon decay : 0.995
- replay buffer size : 50 000

reward function : +1 for each step / -10 if the bird die / +10 if the bird goes through a pipe


Finaly, the learning took me 2 hours and the average score is 99.78.
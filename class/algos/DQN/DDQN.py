"""
Algorithme Double DQN
paper : https://arxiv.org/pdf/1509.06461.pdf
cartpole parameters : https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch
import gym
from collections import deque
import numpy as np
import random

"""
Hyper parameters
"""
GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = BATCH_SIZE
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 200
TARGET_UPDATE_FREQUENCY = 5 # Mise a jour des poids de la fonction target
NB_STEPS = 200000 # nombre d'iterations
TEST_FREQUENCY = 5
LOG_DIR = './log/ddqn_mse_lunarlander'
LR = 2.5e-4

class Network(nn.Module) :

    def __init__(self,env) -> None:
        super().__init__()

        nb_obs = int(np.prod(env.observation_space.shape))
        nb_actions = env.action_space.n
        nb_neurons = 64


        self.net = nn.Sequential(
            nn.Linear(nb_obs, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons,nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons,nb_actions)
        )

    def forward(self,x):
        return self.net(x)

    def act(self, obs):
        obs_t = torch.as_tensor(obs , dtype=torch.float32)
        q_values = self(obs_t.unsqueeze(0))
        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()
        return action

env = gym.make('LunarLander-v2')

replay_buffer = deque(maxlen=BUFFER_SIZE)

online_net = Network(env)
target_net = Network(env)

target_net.load_state_dict(online_net.state_dict())

optimizer = torch.optim.RMSprop(online_net.parameters(),lr=LR)
summary_writer = SummaryWriter(LOG_DIR)


obs = env.reset()

for _ in range(MIN_REPLAY_SIZE) :
    action = env.action_space.sample()

    new_obs , rew , done , _ = env.step(action)

    transition = (obs , action , rew , done , new_obs)
    replay_buffer.append(transition)
    obs = new_obs

    if done : 
        obs = env.reset()

episode = 0
for step in range(NB_STEPS) : 
    epsilon = np.interp(step , [0, EPSILON_DECAY] , [EPSILON_START , EPSILON_END])

    rnd_sample = random.random()

    if rnd_sample <= epsilon :
        action = env.action_space.sample()
    else: 
        action = online_net.act(obs)

    new_obs , rew , done , _ = env.step(action)

    transition = (obs , action , rew , done , new_obs)
    replay_buffer.append(transition)
    obs = new_obs

    transitions = random.sample(replay_buffer , BATCH_SIZE)

    obses = np.asarray([t[0] for t in transitions])
    actions = np.asarray([t[1] for t in transitions])
    rews = np.asarray([t[2] for t in transitions])
    dones = np.asarray([t[3] for t in transitions])
    new_obses = np.asarray([t[4] for t in transitions])

    obses_t = torch.as_tensor(obses, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
    dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
    new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

    targets = rews_t + GAMMA * (1 - dones_t) * torch.gather(input=target_net(new_obses_t), dim=1, index=torch.argmax(online_net(new_obses_t),dim=1).unsqueeze(-1))

    q_values = online_net(obses_t)

    action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

    loss = (action_q_values - targets).pow(2).mean() # pour eviter l'explosion de gradient

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if episode % TARGET_UPDATE_FREQUENCY == 0 :
        target_net.load_state_dict(online_net.state_dict())

    if done : 
        episode += 1
        if episode % TEST_FREQUENCY == 0 :
            state = env.reset()
            done = False
            cum_sum = 0
            while not done :
                state = torch.as_tensor(state,dtype=torch.float32)
                action = torch.argmax(online_net(state)).item()
                new_state,reward,done,_ = env.step(action)
                cum_sum+=reward
                state=new_state
            print(f"episode {episode} - timestep {step} - testreward {cum_sum}")
            summary_writer.add_scalar('Rewards' , cum_sum , global_step=episode)

        obs = env.reset()
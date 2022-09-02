"""
Algorithme Double DQN avec Prioritized Experience Replay
"""
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch
import gym
import collections
from collections import deque
import numpy as np
import random
import typing


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
LOG_DIR = './log/ddqnper_lunarlander'
LR = 2.5e-4
ALPHA = 0.5



"""
Prioritized Experience Replay
"""


_field_names = [
    "state",
    "action",
    "reward",
    "done",
    "next_state"
    
]
Experience = collections.namedtuple("Experience", field_names=_field_names)

class PrioritizedExperienceReplayBuffer:
    def __init__(self,
                 batch_size: int,
                 buffer_size: int,
                 alpha: float = 0.0,
                 random_state: np.random.RandomState = None) -> None:
        self._batch_size = batch_size
        self._buffer_size = buffer_size
        self._buffer_length = 0
        self._buffer = np.empty(self._buffer_size, dtype=[("priority", np.float32), ("experience", Experience)])
        self._alpha = alpha
        self._random_state = np.random.RandomState() if random_state is None else random_state
        
    def __len__(self) -> int:
        return self._buffer_length

    def alpha(self):
        return self._alpha

    def batch_size(self) -> int:
        return self._batch_size
    
    def buffer_size(self) -> int:
        return self._buffer_size

    def add(self, experience: Experience) -> None:
        priority = 1.0 if self.is_empty() else self._buffer["priority"].max()
        if self.is_full():
            if priority > self._buffer["priority"].min():
                idx = self._buffer["priority"].argmin()
                self._buffer[idx] = (priority, experience)
            else:
                pass
        else:
            self._buffer[self._buffer_length] = (priority, experience)
            self._buffer_length += 1

    def is_empty(self) -> bool:
        return self._buffer_length == 0
    
    def is_full(self) -> bool:
        return self._buffer_length == self._buffer_size
    
    def sample(self, beta: float) -> typing.Tuple[np.array, np.array, np.array]:
        """Sample a batch of experiences from memory."""
        ps = self._buffer[:self._buffer_length]["priority"]
        sampling_probs = ps**self._alpha / np.sum(ps**self._alpha)
        idxs = self._random_state.choice(np.arange(ps.size),
                                         size=self._batch_size,
                                         replace=True,
                                         p=sampling_probs)
        
        experiences = self._buffer["experience"][idxs]        
        weights = (self._buffer_length * sampling_probs[idxs])**-beta
        normalized_weights = weights / weights.max()
        
        return idxs, experiences, normalized_weights

    def update_priorities(self, idxs: np.array, priorities: np.array) -> None:
        self._buffer["priority"][idxs] = priorities




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

def exponential_annealing_schedule(episode, rate=1e-2):
    return 1 - np.exp(-rate * episode)

env = gym.make('LunarLander-v2')

replay_buffer = PrioritizedExperienceReplayBuffer(batch_size=BATCH_SIZE,buffer_size=BUFFER_SIZE,alpha=ALPHA)# PrioritizedReplayBuffer(maxlength=BUFFER_SIZE)

online_net = Network(env)
target_net = Network(env)

target_net.load_state_dict(online_net.state_dict())

optimizer = torch.optim.RMSprop(online_net.parameters(),lr=LR)
summary_writer = SummaryWriter(LOG_DIR)



for _ in range(MIN_REPLAY_SIZE) :
    action = env.action_space.sample()

    new_obs , rew , done , _ = env.step(action)

    transition = (obs , action , rew , done , new_obs)
    replay_buffer.add(transition)
    obs = new_obs

    if done : 
        obs = env.reset()


episode = 0
obs = env.reset()
for step in range(NB_STEPS) : 
    epsilon = np.interp(step , [0, EPSILON_DECAY] , [EPSILON_START , EPSILON_END])

    rnd_sample = random.random()

    if rnd_sample <= epsilon :
        action = env.action_space.sample()
    else: 
        action = online_net.act(obs)

    new_obs , rew , done , _ = env.step(action)

    transition = (obs , action , rew , done , new_obs)
    replay_buffer.add(transition)
    obs = new_obs

    beta = exponential_annealing_schedule(episode)
            
    idxs, experiences, normalized_weights = replay_buffer.sample(beta)

    _sampling_weights = (torch.Tensor(normalized_weights).view((-1, 1)))

    obses_t, actions_t, rews_t, dones_t ,new_obses_t  = (torch.Tensor(vs) for vs in zip(*experiences))

    actions_t = actions_t.long().unsqueeze(dim=1)
    rews_t = rews_t.unsqueeze(dim=1)
    dones_t = dones_t.unsqueeze(dim=1)

    targets = rews_t + GAMMA * (1 - dones_t) * torch.gather(input=target_net(new_obses_t), dim=1, index=torch.argmax(online_net(new_obses_t),dim=1).unsqueeze(-1))

    q_values = online_net(obses_t)

    action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

    td_error = targets - action_q_values

    replay_buffer.update_priorities(idxs, td_error.abs().cpu().detach().numpy().flatten())

    loss = torch.mean((td_error * _sampling_weights)**2) 

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
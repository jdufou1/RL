"""
Double DQN with Prioritized Experience Replay for Flappy Bird Environment
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
import flappy_bird_gym
import cv2

GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 10000
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 200
TARGET_UPDATE_FREQUENCY = 10000 # Mise a jour des poids de la fonction target
NB_STEPS = 10000000 # nombre d'iterations
TEST_FREQUENCY = 5
LOG_DIR = './log/ddqnper_flappy_bird'
LR = 2.5e-4
ALPHA = 0.5
DEVICE = torch.device('cuda:0'  if torch.cuda.is_available() else "cpu")


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

class Env : 

    def __init__(self) -> None:
        pass

    def reset(self) :
        pass

    def step(self , action) :
        pass

    def reward_function():
        pass

    def transition_function():
        pass

class EnvFlappyBird(Env) : 

    def __init__(self, env) -> None:
        super().__init__()
        self.env = env

    def reset(self):
        return self.env.reset()

    def step(self, action) :
        stp1 , reward ,done , probability = self.env.step(action)
        return stp1 , self.reward_function(stp1 , reward ,done , probability) , done , self.transition_function( stp1 , reward ,done , probability)

    def reward_function(self, stp1 , reward ,done , probability):
        if done :
            return -1000
        else :
            return 1

    def transition_function(self, stp1 , reward ,done , probability):
        return probability


class Network(nn.Module) :

    def __init__(self,env) -> None:
        super().__init__()

        self.nb_actions = 2

        self.net =  nn.Sequential(
            nn.Conv2d(4,32,8,stride = 4,padding=(0,0)),
            nn.MaxPool2d(kernel_size=2 , padding=(0,0)),
            nn.ReLU(),
            nn.Conv2d(32,64,4,stride = 2 , padding = (1,1)),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2, stride = 2,padding = (1,1)),
            nn.ReLU(),
            nn.Conv2d(64,64,3,stride = 1, padding = (1,1)),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2, padding = (1,1)),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.nb_actions)
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

env_gym = flappy_bird_gym.make('FlappyBird-rgb-v0')
env = EnvFlappyBird(env_gym)
replay_buffer = PrioritizedExperienceReplayBuffer(batch_size=BATCH_SIZE,buffer_size=BUFFER_SIZE,alpha=ALPHA)# PrioritizedReplayBuffer(maxlength=BUFFER_SIZE)

online_net = Network(env)
target_net = Network(env)

target_net.load_state_dict(online_net.state_dict())

optimizer = torch.optim.RMSprop(online_net.parameters(),lr=LR)
summary_writer = SummaryWriter(LOG_DIR)


last_rewards = deque(maxlen=100)

state = env.reset()
state_converted = cv2.cvtColor(cv2.resize(state, (80, 80)), cv2.COLOR_BGR2GRAY)
state_stacked = np.stack((state_converted, state_converted, state_converted, state_converted), axis=2)
state_stacked_t = torch.as_tensor(state_stacked , device = DEVICE, dtype = torch.float32).unsqueeze(0).transpose(3 , 1)
for _ in range(MIN_REPLAY_SIZE) :

    action = random.randint(0,1)

    new_state , rew , done , _ = env.step(action)

    new_state_converted = cv2.cvtColor(cv2.resize(new_state, (80, 80)), cv2.COLOR_BGR2GRAY)
    new_state_converted = new_state_converted.reshape(new_state_converted.shape[0],new_state_converted.shape[1],1 )
    new_state_stacked = np.append(new_state_converted, state_stacked[:, :, :3], axis = 2)
    new_state_stacked_t = torch.as_tensor(new_state_stacked , device = DEVICE, dtype = torch.float32).unsqueeze(0).transpose(3 , 1)

    transition = (state_stacked_t , action , rew , done , new_state_stacked_t)
    replay_buffer.add(transition)
    state_stacked_t = new_state_stacked_t

    if done : 
        state = env.reset()
        state_converted = cv2.cvtColor(cv2.resize(state, (80, 80)), cv2.COLOR_BGR2GRAY)
        state_stacked = np.stack((state_converted, state_converted, state_converted, state_converted), axis=2)
        state_stacked_t = torch.as_tensor(state_stacked , device = DEVICE, dtype = torch.float32).unsqueeze(0).transpose(3 , 1)


episode = 0
state = env.reset()
state_converted = cv2.cvtColor(cv2.resize(state, (80, 80)), cv2.COLOR_BGR2GRAY)
state_stacked = np.stack((state_converted, state_converted, state_converted, state_converted), axis=2)
state_stacked_t = torch.as_tensor(state_stacked , device = DEVICE, dtype = torch.float32).unsqueeze(0).transpose(3 , 1)

epsilon = EPSILON_START
for step in range(NB_STEPS) : 
    if epsilon > EPSILON_END :
        epsilon -= (EPSILON_START - EPSILON_END) / NB_STEPS

    rnd_sample = random.random()

    if rnd_sample <= epsilon :
        action = random.randint(0,1)
    else: 
        action = torch.argmax(online_net(state_stacked_t)).item()

    new_state , rew , done , _ = env.step(action)

    new_state_converted = cv2.cvtColor(cv2.resize(new_state, (80, 80)), cv2.COLOR_BGR2GRAY)
    new_state_converted = new_state_converted.reshape(new_state_converted.shape[0],new_state_converted.shape[1],1 )
    new_state_stacked = np.append(new_state_converted, state_stacked[:, :, :3], axis = 2)
    new_state_stacked_t = torch.as_tensor(new_state_stacked , device = DEVICE, dtype = torch.float32).unsqueeze(0).transpose(3 , 1)

    transition = (state_stacked_t , action , rew , done , new_state_stacked_t)
    
    replay_buffer.add(transition)

    state_stacked_t = new_state_stacked_t

    beta = exponential_annealing_schedule(episode)
            
    idxs, experiences, normalized_weights = replay_buffer.sample(beta)

    _sampling_weights = (torch.Tensor(normalized_weights).view((-1, 1)))

    obses, actions, rews, dones ,new_obses  = (vs for vs in zip(*experiences))

    obses_t = torch.stack([o for o in obses]).squeeze()
    actions_t = torch.as_tensor(np.array([a for a in actions]))
    rews_t = torch.as_tensor(np.array([r for r in rews]))
    dones_t = torch.as_tensor(np.array([o for o in dones] , dtype = int) , dtype=torch.int)
    new_obses_t = torch.stack([o for o in new_obses]).squeeze()

    # obses_t = torch.as_tensor(obses,dtype=torch.float32)

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

    if step % TARGET_UPDATE_FREQUENCY == 0 :
        target_net.load_state_dict(online_net.state_dict())
        torch.save(online_net.state_dict(), "flappy_bird_onlinenet_weights")

    if done : 
        episode += 1
        if episode % TEST_FREQUENCY == 0 :
            state = env.reset()
            state_converted = cv2.cvtColor(cv2.resize(state, (80, 80)), cv2.COLOR_BGR2GRAY)
            state_stacked = np.stack((state_converted, state_converted, state_converted, state_converted), axis=2)
            state_stacked_t = torch.as_tensor(state_stacked , device = DEVICE, dtype = torch.float32).unsqueeze(0).transpose(3 , 1)
            done = False
            cum_sum = 0
            while not done :
                action = torch.argmax(online_net(state_stacked_t)).item()
                new_state,reward,done,_ = env.step(action)
                new_state_converted = cv2.cvtColor(cv2.resize(new_state, (80, 80)), cv2.COLOR_BGR2GRAY)
                new_state_converted = new_state_converted.reshape(new_state_converted.shape[0],new_state_converted.shape[1],1 )
                new_state_stacked = np.append(new_state_converted, state_stacked[:, :, :3], axis = 2)
                new_state_stacked_t = torch.as_tensor(new_state_stacked , device = DEVICE, dtype = torch.float32).unsqueeze(0).transpose(3 , 1)
                
                cum_sum+=reward
                state_stacked_t=new_state_stacked_t
            last_rewards.append(cum_sum)
            average_last_rewards = sum(last_rewards) / len(last_rewards)
            print(f"episode {episode} - timestep {step} - testreward {cum_sum} - average_last_rewards {average_last_rewards} - eps {epsilon}")
            summary_writer.add_scalar('Rewards' , cum_sum , global_step=step)
            summary_writer.add_scalar('average_last_rewards' , average_last_rewards , global_step=step)

        state = env.reset()
        state_converted = cv2.cvtColor(cv2.resize(state, (80, 80)), cv2.COLOR_BGR2GRAY)
        state_stacked = np.stack((state_converted, state_converted, state_converted, state_converted), axis=2)
        state_stacked_t = torch.as_tensor(state_stacked , device = DEVICE, dtype = torch.float32).unsqueeze(0).transpose(3 , 1)
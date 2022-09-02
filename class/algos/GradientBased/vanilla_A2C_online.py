"""
TME 5 : Policy Gradient
One Step Actor-Critic
"""

"""
Import
"""
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from torch import nn
import gym

"""
Hyper parameters
"""

LR_ACTOR = 0.001
LR_CRITIC = 0.001
GAMMA = 0.99
NB_EPISODE = 1000
LOG_INTERVAL = 5
LOG_DIR = './log/vanilla_actor_critic_online_carpole'

class Network_Critic(nn.Module) : 

    def __init__(self,env) -> None:
        super().__init__()
        self.env = env
        self.nb_obses = int(np.prod(env.observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(self.nb_obses , 64),
            nn.Tanh(),
            nn.Linear(64,1)
        )

    def forward(self,x) :
        return self.net(x)

class Network_Actor(nn.Module) : 

    def __init__(self,env) -> None:
        super().__init__()
        self.env = env
        self.nb_actions = env.action_space.n
        self.nb_obses = int(np.prod(env.observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(self.nb_obses , 64),
            nn.Tanh(),
            nn.Linear(64,self.nb_actions),
            nn.Softmax()
        )

    def forward(self,x) :
        return self.net(x)

    def act(self,obs) :
        obs_t = torch.as_tensor(obs , dtype=torch.float32)
        probs = self.net(obs_t)
        if obs_t.shape[0] == 4 :
            probs = probs.unsqueeze(0)

        list_prob = torch.stack([probs[i][torch.argmax(probs,dim = 1)[i]] for i in range(probs.shape[0])])

        log_prob = torch.log(list_prob)
        return [torch.argmax(probs,dim = 1)[i].detach().item() for i in range(probs.shape[0])], log_prob

# load environment
env = gym.make('CartPole-v0')

summary_writer = SummaryWriter(LOG_DIR)

policy_net = Network_Actor(env)
value_net = Network_Critic(env)

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=LR_ACTOR)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=LR_CRITIC)

episode_counter = 0

# print("log list : ",log_list)

# main loop

with torch.autograd.set_detect_anomaly(mode=True):

    while episode_counter < NB_EPISODE :

        reward_sum = 0.0
        step_counter = 0.0

        state = env.reset()
        i = 1
        done = False
        
        while not done :

            # 1. sampling

            # select an action
            action,log_proba = policy_net.act(state)
        
            state_, rew,  done, info = env.step(action[0])

            rew_t = torch.as_tensor(rew, dtype = torch.float32).unsqueeze(-1)
            obs_t = torch.as_tensor(state, dtype = torch.float32)
            done_t = torch.as_tensor(done, dtype = torch.float32).unsqueeze(-1)
            new_obs_t = torch.as_tensor(state_, dtype = torch.float32)

            reward_sum += rew


            huber_loss = nn.HuberLoss()
            target  = rew_t + GAMMA * value_net.forward(new_obs_t)
            loss_value = huber_loss(target, value_net.forward(obs_t))

            # Gradient descent
            optimizer_value.zero_grad()
            loss_value.backward()
            optimizer_value.step()

            # # 3. evaluate advantage function
            with torch.no_grad():
                advantage = rew_t + (GAMMA * done_t * value_net.forward(new_obs_t)) - value_net.forward(obs_t)

            # 4. fit policy
            loss_policy = - log_proba * advantage
            
            # gradient descent
            optimizer_policy.zero_grad()
            loss_policy.backward()
            optimizer_policy.step()

            state = state_

            step_counter += 1

        summary_writer.add_scalar('Rewards',reward_sum,global_step = episode_counter)

        if episode_counter % LOG_INTERVAL == 0 :
            print(f"episode {episode_counter} - rewards : {reward_sum}")

        
        episode_counter += 1
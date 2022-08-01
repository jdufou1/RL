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
LOG_DIR = './log/vanilla_actor_critic_batch_carpole'
BATCH_SIZE = 32

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

# main loop

while episode_counter < NB_EPISODE :

    transitions = list()
    obs = env.reset()

    # sampling in respect to the current policy => on-policy

    for _ in range(BATCH_SIZE):
        action,log_proba = policy_net.act(obs)

        new_obs, rew, done, _ = env.step(action[0])
        transition = (obs, action, rew, done, new_obs,log_proba)
        transitions.append(transition)

        obs = new_obs
        if done:
            obs = env.reset()

    # learning

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
    log_probs = torch.stack([t[5] for t in transitions])
    
    # Compute the critic loss 
        
    huber_loss = nn.HuberLoss()
    target  = rews_t + GAMMA * value_net.forward(new_obses_t)

    loss_value = huber_loss(target, value_net.forward(obses_t))

    # Gradient descent
    optimizer_value.zero_grad()
    loss_value.backward()
    optimizer_value.step()

    # # 3. evaluate advantage function
    with torch.no_grad():
        advantage = rews_t + (GAMMA * dones_t * value_net.forward(new_obses_t)) - value_net.forward(obses_t)

    # 4. Compute the actor loss 
    loss_policy = - log_probs * advantage
    
    # gradient descent
    optimizer_policy.zero_grad()
    loss_policy.sum().backward()
    optimizer_policy.step()

    # Test

    if episode_counter % LOG_INTERVAL == 0 :
        rewards = 0.0
        obs = env.reset()
        done = False
        while not done :
            action,log_proba = policy_net.act(obs)
            new_obs, rew, done, _ = env.step(action[0])
            rewards += rew
            obs = new_obs
            
        print(f"episode {episode_counter} - test reward : {rewards}")
        summary_writer.add_scalar('Rewards',rewards,global_step = episode_counter)

    episode_counter += 1

   


        
        
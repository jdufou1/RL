import torch
import torch.nn as nn
import gym
import numpy as np
import random
from collections import deque
from tqdm import tqdm

env = gym.make("Walker2d-v2")

NB_ITERATION = 1e6

REPLAY_BUFFER_SIZE = 10000
NB_ITERATION = 10000
FREQUENCY_UPDATE = 4
FREQUENCY_TEST = 100
BATCH_SIZE = 64
DISCOUNT_FACTOR = 0.99
TAU = 0.001
actor_learning_rate = 0.0001
critic_learning_rate = 0.001

list_res = []
nb_actions = 6
nb_observations = 17

class PolicyNetwork(nn.Module) : 

    def __init__(self,
                nb_actions,
                nb_observations,
                nb_neurons=40
                ) -> None:

        super().__init__()

        self.nb_actions = nb_actions
        self.nb_observations = nb_observations

        self.net = nn.Sequential(
            nn.Linear(nb_observations,50),
            nn.ReLU(),
            nn.Linear(50,20),
            nn.ReLU(),
            nn.Linear(20,20),
            nn.Tanh(),
            nn.Linear(20,nb_actions),
            nn.Tanh()
        )

    def forward(self,x) :
        return self.net(x)


class CriticNetwork(nn.Module) :

    def __init__(self,
                nb_actions,
                nb_observations,
                nb_neurons=600
                ) -> None:

        super().__init__()

        self.nb_actions = nb_actions
        self.nb_observations = nb_observations

        self.net = nn.Sequential(
            nn.Linear(nb_observations + nb_actions,50),
            nn.ReLU(),
            nn.Linear(50,20),
            nn.ReLU(),
            nn.Linear(20,20),
            nn.ReLU(),
            nn.Linear(20,1),
        )

    def forward(self,x) :
        return self.net(x)


replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

policy_network = PolicyNetwork(nb_actions,nb_observations,200)
critic_network = CriticNetwork(nb_actions,nb_observations,200)

policy_network_target = PolicyNetwork(nb_actions,nb_observations,200)
critic_network_target = CriticNetwork(nb_actions,nb_observations,200)

policy_network_target.load_state_dict(policy_network.state_dict())
critic_network_target.load_state_dict(critic_network.state_dict())


optimizer_policy = torch.optim.Adam(policy_network.parameters(), lr=actor_learning_rate)
optimizer_critic = torch.optim.Adam(critic_network.parameters(), lr=critic_learning_rate)



def test() : 
    state = env.reset()
    done = False
    cum_sum = 0
    while not done :
        state_t = torch.as_tensor(state, dtype = torch.float32)
        action = policy_network(state_t)
        action = torch.clip(action , min=-1 , max=1).detach().numpy()
        new_state,reward,done,_ = env.step(action)
        state = new_state
        cum_sum += reward
    return cum_sum


for i in tqdm(range(NB_ITERATION)) :

   

    state = env.reset()
    state_t = torch.as_tensor(state, dtype = torch.float32)
    
    action = policy_network(state_t) + torch.rand(nb_actions)
    
    action = torch.clip(action , min=-1 , max=1).detach().numpy()
    
    new_state,reward,done,_ = env.step(action)
    transition = (state,action,reward,new_state,done)
    
    replay_buffer.append(transition)

    if done : 
        state = env.reset()
    else : 
        state = new_state

    if i % FREQUENCY_UPDATE == 0 and len(replay_buffer) >= BATCH_SIZE:

        batch = random.sample(replay_buffer , BATCH_SIZE)

        states = np.asarray([exp[0] for exp in batch] , dtype=np.float32)
        actions = np.asarray([exp[1] for exp in batch], dtype=int)
        rewards = np.asarray([exp[2] for exp in batch], dtype=np.float32)
        new_states = np.asarray([exp[3] for exp in batch], dtype=np.float32)
        dones = np.asarray([exp[4] for exp in batch], dtype=int)

        states_t = torch.as_tensor(states , dtype=torch.float32)
        actions_t = torch.as_tensor(actions , dtype=torch.int64)
        rewards_t = torch.as_tensor(rewards , dtype=torch.float32)
        new_states_t = torch.as_tensor(new_states , dtype=torch.float32)
        dones_t = torch.as_tensor(dones , dtype=torch.int64)

        
        new_actions_t = policy_network_target(new_states_t)
        new_actions_t = torch.clip(new_actions_t , min=-1 , max=1)
        
        """
        print("new_states_t : ",new_states_t.shape)
        print("new_actions_t : ",new_actions_t.shape)
        print("torch.cat(new_states_t , new_actions_t)" ,torch.cat((new_states_t , new_actions_t),dim=1).shape)
        """
        
        targets = (rewards_t + 
                DISCOUNT_FACTOR * 
                (1 - dones_t) * 
                critic_network_target(torch.cat((new_states_t , new_actions_t),dim=1))
                ).detach()

        mse = torch.nn.MSELoss()
        loss_critic = mse(critic_network(torch.cat((states_t , actions_t),dim=1)) , targets)

        optimizer_critic.zero_grad()
        loss_critic.backward()
        optimizer_critic.step()

        # print(critic_network(torch.cat((states_t , torch.clip(policy_network(states_t) , min=-1 , max=1)),dim=1)).shape)
        
        loss_policy = - critic_network(torch.cat((states_t , torch.clip(policy_network(states_t) , min=-1 , max=1)),dim=1)).mean()

        optimizer_policy.zero_grad()
        loss_policy.backward()
        optimizer_policy.step()

        state_dict_critic_network_target = critic_network_target.state_dict()
        state_dict_critic_network = critic_network.state_dict()

        state_dict_policy_network_target = policy_network_target.state_dict()
        state_dict_policy_network = policy_network.state_dict()

        for (_, param_target),(_, param) in zip(state_dict_critic_network_target.items(),state_dict_critic_network.items()):
            transformed_param = TAU * param_target + (1 - TAU) * param
            param_target.copy_(transformed_param)


        for (_, param_target),(_, param) in zip(state_dict_policy_network_target.items(),state_dict_policy_network.items()):
            transformed_param = TAU * param_target + (1 - TAU) * param
            param_target.copy_(transformed_param)


    if i % 10 == 0 :
        res = test()
        list_res.append(res)
        #print("iteration : ",i,"test reward : ",res)
        
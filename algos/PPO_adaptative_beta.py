"""
PPO with adaptative beta
"""

import gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm

class NetworkActor(nn.Module) : 
    
    def __init__(self,
                 nb_obs,
                 nb_actions,
                 nb_neurons
                ) :
        super().__init__()
        self.nb_obs = nb_obs
        self.nb_actions = nb_actions
        self.nb_neurons = nb_neurons
        
        self.net = nn.Sequential(
            nn.Linear(self.nb_obs,self.nb_neurons),
            nn.ReLU(),
            nn.Linear(self.nb_neurons,self.nb_neurons),
            nn.ReLU(),
            nn.Linear(self.nb_neurons,self.nb_actions),
            nn.Softmax()
        )
        
    def forward(self,x) :
        return self.net(x)


class NetworkCritic(nn.Module) : 
    
    def __init__(self,
                 nb_obs,
                 nb_actions,
                 nb_neurons
                ) :
        super().__init__()
        self.nb_obs = nb_obs
        self.nb_actions = nb_actions
        self.nb_neurons = nb_neurons
        
        
        self.net = nn.Sequential(
            nn.Linear(self.nb_obs,self.nb_neurons),
            nn.ReLU(),
            nn.Linear(self.nb_neurons,self.nb_neurons),
            nn.ReLU(),
            nn.Linear(self.nb_neurons,1),
            
        )
        
    def forward(self,x) :
        return self.net(x)



def kl_div(p, q):
    
    p =  torch.distributions.utils.clamp_probs(p)
    q =  torch.distributions.utils.clamp_probs(q)
    return (p * (torch.log(p) - torch.log(q))).sum(-1)


class AgentPPO :
    
    def __init__(self,
                kl_value : float,
                discount_factor : float,
                learning_rate : float,
                timesteps_per_batch : int,
                max_timesteps_per_episode : int,
                nb_eval : int,
                test_frequency : float,
                network_actor : NetworkActor,
                network_critic : NetworkCritic,
                optimizer_actor,
                optimizer_critic,
                env
                ) -> None:
        
        self.kl_value = kl_value
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.timesteps_per_batch = timesteps_per_batch
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.nb_eval = nb_eval
        self.test_frequency = test_frequency
        self.network_actor = network_actor
        self.network_critic = network_critic
        self.optimizer_actor = optimizer_actor
        self.optimizer_critic = optimizer_critic
        self.env = env
        
        self.beta = 1.0
        
    def collect(self):
        batch_episode = list()
        batch_experiences = list()
        
        size_batch = 0
        
        while size_batch < self.timesteps_per_batch :
            i = 0
            done = False
            state = self.env.reset()
            while not done or i < self.max_timesteps_per_episode :
                state_t = torch.as_tensor(state , dtype=torch.float32)
                best_action = np.random.choice(a=np.arange(2),p=self.network_actor(state_t).detach().numpy().reshape(2)) # torch.argmax(self.network_actor(state_t)).item()
                new_state,reward,done,_ = self.env.step(best_action)
                experience = (state,reward,done,new_state,best_action)
                batch_experiences.append(experience)
                state = new_state
                i += 1
                size_batch += 1
                
            batch_episode.append(batch_experiences)
            batch_experiences = list()
                
        return batch_episode
            
    
    def compute_advantage(self,batch_episode) :
        batch_advantage = list()
        for episode in batch_episode: 
            
            discounted_reward = 0
            
            for (state,reward,done,new_state,best_action) in reversed(episode) :
                
                state_t = torch.as_tensor(state , dtype=torch.float32)
                new_state_t = torch.as_tensor(new_state , dtype=torch.float32)
                
                discounted_reward = reward + discounted_reward * self.discount_factor
                advantage = reward + self.discount_factor * (1 - done) * self.network_critic(new_state_t) - self.network_critic(state_t)
                advantage = advantage.detach().numpy()
                
                new_data = (state,advantage,discounted_reward,new_state,best_action,reward,done)
                batch_advantage.append(new_data)
                
        return batch_advantage
    
    def step(self) : 
        batch_episode = self.collect()
        batch_advantage = self.compute_advantage(batch_episode)

        states = np.asarray([exp[0] for exp in batch_advantage],dtype=np.float32)
        states_t = torch.as_tensor(states, dtype=torch.float32)
        
        advantages = np.asarray([exp[1] for exp in batch_advantage],dtype=np.float32)
        advantages_t = torch.as_tensor(advantages , dtype=torch.float32)
        
        advantages_t = (advantages_t - advantages_t.mean()) / advantages_t.std()
        
        rtgs = np.asarray([rtg[2] for rtg in batch_advantage],dtype=np.float32)
        rtgs_t = torch.as_tensor(rtgs, dtype=torch.float32)
        
        actions = np.asarray([exp[4] for exp in batch_advantage],dtype=np.int64)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(1)

        first_proba = self.network_actor(states_t).detach()
        first_distrib = torch.gather(input=first_proba,dim=1,index=actions_t).detach()
        
        for _ in range(5) :
            
            # Surrogate
            surrogate_loss = (torch.gather(input=torch.distributions.utils.clamp_probs(self.network_actor(states_t)),dim=1,index=actions_t) / first_distrib) * advantages_t.detach()# torch.gather(input=old_network_actor(states_t).detach(),dim=1,index=actions_t)) * advantages_t
            surrogate_loss = torch.mean(surrogate_loss)

            # kullback_leibler
            kullback_leibler = kl_div(self.network_actor(states_t),first_proba) # kl_div(old_network_actor(states_t), self.network_actor(states_t))
            kullback_leibler = torch.mean(kullback_leibler)

            loss_actor = - (surrogate_loss - (self.beta * kullback_leibler) )

            self.optimizer_actor.zero_grad()
            loss_actor.backward()
            self.optimizer_actor.step()
        
        dkl = kl_div(self.network_actor(states_t),first_proba)
        dkl = torch.mean(dkl)
        
        if dkl >= 1.5 * self.kl_value :
            self.beta = 2.0 * self.beta
            
        if dkl <= self.kl_value / 1.5 :
            self.beta = 0.5 * self.beta
            
        V = self.network_critic(states_t)
 
        mse = torch.nn.MSELoss()
        loss_critic = mse(rtgs_t.unsqueeze(1),V)
        
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()
        
        
    def test(self) :
        """
        retourne la reward cumulée de l'episode
        """
        state = self.env.reset()
        cum_rew = 0
        done = False
        while not done :
            state_t = torch.as_tensor(state, dtype=torch.float32)
            best_action = np.random.choice(a=np.arange(2),p=self.network_actor(state_t).detach().numpy().reshape(2)) # torch.argmax(self.network_actor(state_t)).item()
            new_state,reward,done,_ = self.env.step(best_action)
            cum_rew += reward
            state = new_state
        return cum_rew
        

    def learning(self): 
        list_rewards = list()
        for i in tqdm(range(self.nb_eval)):
            self.step()
            if i % self.test_frequency == 0:
                reward = self.test()
                list_rewards.append(reward)
                # print(f"iteration : {i} - test reward : {reward} - beta : {self.beta}")
        return list_rewards


env = gym.make("CartPole-v0")

init_network_actor = NetworkActor(
    nb_obs = 4,
    nb_actions = 2,
    nb_neurons = 60
)

init_network_critic = NetworkCritic(
    nb_obs = 4,
    nb_actions = 2,
    nb_neurons = 60
)

optimizer_actor = torch.optim.Adam(init_network_actor.parameters(), lr=0.005)
optimizer_critic = torch.optim.Adam(init_network_critic.parameters(), lr=0.005)

agentPPO = AgentPPO(
    kl_value = 0.05,
    discount_factor = 0.99,
    learning_rate = 0.005,
    timesteps_per_batch =  1000,# 4800,
    max_timesteps_per_episode = 200,# 1600,
    nb_eval = 500,
    test_frequency = 1,
    network_actor = init_network_actor,
    network_critic = init_network_critic,
    optimizer_actor = optimizer_actor,
    optimizer_critic = optimizer_critic,
    env = env
)


list_rewards = agentPPO.learning()

plt.figure()
plt.title("Cartpole-v0 PPO")
plt.xlabel("iteration")
plt.ylabel("rewards")
plt.plot(list_rewards)
plt.show()
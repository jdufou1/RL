import gym
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from collections import deque
import time



# env = gym.make("LunarLander-v2")
# nb_actions = 4
# nb_observations = 8

env = gym.make("CartPole-v0")
nb_actions = 2
nb_observations = 4

nb_episode = 1

discount_factor = 0.99
learning_rate = 2e-3
test_frequency = 10
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.02
batch_size = 64
size_replay_buffer = int(1e5)
update_frequency = 1
tau = 1e-3 


def test(q_network) :
    
    state = env.reset()
    done = False
    cum_sum = 0
    while not done :
        state_t = torch.as_tensor(state , dtype = torch.float32).unsqueeze(0)
        action = torch.argmax(q_network(state_t)).item()
        new_state,reward,done,_ = env.step(action)
        state = new_state
        cum_sum += reward
        
    return cum_sum



class DuelingQNetwork(nn.Module) :
    
    def __init__(self,
              nb_actions,
              nb_observations) : 
        
        super().__init__()
        self.nb_actions = nb_actions
        self.nb_observations = nb_observations
        
        self.net = nn.Sequential(
            nn.Linear(nb_observations,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,32)
        )
        
        self.net_advantage = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32,nb_actions)
        )
        
        self.net_state_value = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32,1)
        )
        
    def advantage(self,x) :
        return self.net_advantage(self.net(x))
    
    def state_value(self,x) :
        return self.net_state_value(self.net(x))
    
    def forward(self,x) :
        return self.state_value(x) + self.advantage(x) - torch.mean(self.advantage(x),dim=1).unsqueeze(1)





replay_buffer = deque(maxlen=size_replay_buffer)
q_network = DuelingQNetwork(nb_actions,nb_observations)
q_target_network = DuelingQNetwork(nb_actions,nb_observations)
q_target_network.load_state_dict(q_network.state_dict())
optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)
list_tests_6 = []
timestep = 0

bestModel = DuelingQNetwork(nb_actions,nb_observations)
bestModel.load_state_dict(q_network.state_dict())
bestvalue = -1e9

average_list = deque(maxlen=100)



def play_env(env,fps=-1):
    """
        Play an episode :
        * agent : agent with two functions : act(state) -> action, and store(state,action,state,reward)
        * max_ep : maximal length of the episode
        * fps : frame per second,not rendering if <=0
        * verbose : True/False print debug messages
        * return the cumulative reward
    """
    state = env.reset()
    done = False
    while not done :
        state_t = torch.as_tensor(state , dtype = torch.float32).unsqueeze(0)
        action = torch.argmax(bestModel(state_t)).item()
        new_state,_,done,_ = env.step(action)
        if fps>0:
            env.render()
            time.sleep(1/fps)
        state = new_state


for episode in tqdm(range(nb_episode)) :
    state = env.reset()
    done = False
    
    cumul = 0
    epsilon = max(epsilon * epsilon_decay,epsilon_min)
    
    while not done : 
        state_t = torch.as_tensor(state , dtype = torch.float32).unsqueeze(0)
        
        if random.random() < epsilon :
            action = torch.argmax(q_network(state_t)).item()
        else :
            action = env.action_space.sample()
            
        new_state,reward,done,_ = env.step(action)

        cumul += reward
        
        transition = (state,action,done,reward,new_state)
        replay_buffer.append(transition)
        
        if len(replay_buffer) >= batch_size and timestep % update_frequency == 0 :
        
            batch = random.sample(replay_buffer,batch_size)

            states = np.asarray([exp[0] for exp in batch],dtype=np.float32)
            actions = np.asarray([exp[1] for exp in batch],dtype=int)
            dones = np.asarray([exp[2] for exp in batch],dtype=int)
            rewards = np.asarray([exp[3] for exp in batch],dtype=np.float32)
            new_states = np.asarray([exp[4] for exp in batch],dtype=np.float32)
            
            states_t = torch.as_tensor(states , dtype=torch.float32)
            dones_t = torch.as_tensor(dones , dtype = torch.int64).unsqueeze(1)
            new_states_t = torch.as_tensor(new_states , dtype=torch.float32)
            actions_t = torch.as_tensor(actions , dtype = torch.int64).unsqueeze(1)
            rewards_t = torch.as_tensor(rewards , dtype=torch.float32).unsqueeze(1)
            
            
            y_target = rewards_t + discount_factor * (1 - dones_t) * torch.gather(q_target_network(new_states_t),dim=1,index=torch.argmax(q_network(new_states_t),dim=1).unsqueeze(1)).detach()

            mse = nn.MSELoss()

            loss = mse(torch.gather(q_network(states_t),dim=1,index=actions_t), y_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            for target_param, local_param in zip(q_target_network.parameters(), q_network.parameters()):
                target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)  
                
        timestep += 1
        
        state = new_state
        
    
    average_list.append(cumul)
    
    if episode % test_frequency == 0 :
        t =  0
        for _ in range(10) :
            t += test(q_network)
        t /= 10
        if t > bestvalue :
            bestvalue = t
            bestModel.load_state_dict(q_network.state_dict())
        avg = sum(average_list) / len(average_list)
        print(f"episode {episode} - test reward : {t} - avg : {avg} - epsilon {epsilon}")
        list_tests_6.append(t)



input()
play_env(env,fps=30)
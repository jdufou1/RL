"""
Model Training for flappy bird 
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random
from collections import deque
from collections import namedtuple
import flappy_bird_gym
import cv2
import time

from torch.utils.tensorboard import SummaryWriter

from numba import cuda

cuda.select_device(0)
cuda.close()
cuda.select_device(0)

LOG_DIR = './log/d3qn_fb_2'

def test(q_network,env) :
    state = env.reset()
    state = np.append(state, env.get_env()._game.player_vel_y)
    state = np.append(state, env.get_env()._game.player_rot)
    done = False
    cum_sum = 0
    timestep = 0
    while not done  :
        state_t = torch.as_tensor(state , dtype=torch.float32, device = DEVICE).unsqueeze(0)
        action = torch.argmax(q_network(state_t)).item()
        new_state,reward,done,_ = env.step(action)
        new_state = np.append(new_state, env.get_env()._game.player_vel_y)
        new_state = np.append(new_state, env.get_env()._game.player_rot)
        state = new_state
        cum_sum += reward   
        timestep +=1
        
    return cum_sum



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
        self.score = 0

    def reset(self):
        self.score = 0
        return self.env.reset()

    def step(self, action) :
        stp1 , reward ,done , probability = self.env.step(action)
        return stp1 , self.reward_function(stp1 , reward ,done , probability) , done , self.transition_function( stp1 , reward ,done , probability)

    def reward_function(self, stp1 , reward ,done , info):
        if done :
            return -10
        else :
            if info["score"] > self.score :
                self.score = info["score"]
                return 10
            else: 
                return 1
    def get_env(self) : 
        return self.env

    def transition_function(self, stp1 , reward ,done , probability):
        return probability


class DuelingQNetwork(nn.Module) :
    
    def __init__(self,
              nb_actions,
              nb_observations) : 
        
        super().__init__()
        self.nb_actions = nb_actions
        self.nb_observations = nb_observations
        
        self.net =  nn.Sequential(
            nn.Linear(nb_observations,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64, 64)
            )
        
        self.net_advantage = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, self.nb_actions)
        )
        
        self.net_state_value = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64,1)
        )
        
    def advantage(self,x) :
        return self.net_advantage(self.net(x))
    
    def state_value(self,x) :
        return self.net_state_value(self.net(x))
    
    def forward(self,x) :
        return self.state_value(x) + self.advantage(x) - torch.mean(self.advantage(x),dim=1).unsqueeze(1)


env_gym = flappy_bird_gym.make('FlappyBird-v0')
env = EnvFlappyBird(env_gym)
nb_actions = 2
nb_observations = 4


nb_episode = 100000

discount_factor = 0.99
learning_rate = 2e-3
test_frequency = 10
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.02
batch_size = 128
size_replay_buffer = 50000
update_frequency = 1
tau = 1e-3  

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

print("Device : ",DEVICE)

summary_writer = SummaryWriter(LOG_DIR)

replay_buffer = deque(maxlen=size_replay_buffer)
q_network = DuelingQNetwork(nb_actions,nb_observations).to(DEVICE)
q_target_network = DuelingQNetwork(nb_actions,nb_observations).to(DEVICE)
q_target_network.load_state_dict(q_network.state_dict())
optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)
list_tests = []
timestep = 0

average_list = deque(maxlen=100)

best_value = -1e9
best_model = DuelingQNetwork(nb_actions,nb_observations).to(DEVICE)

checkpoint_time = time.time()
for episode in tqdm(range(nb_episode)) :
    
    state = env.reset()
    state = np.append(state, env.get_env()._game.player_vel_y)
    state = np.append(state, env.get_env()._game.player_rot)

    done = False
    cumul = 0
    epsilon = max(epsilon * epsilon_decay,epsilon_min)
    
    while not done : 
        state_t = torch.as_tensor(state , dtype=torch.float32, device = DEVICE).unsqueeze(0)
        if random.random() > epsilon :
            action = torch.argmax(q_network(state_t)).item()
        else :
            action = random.randint(0,1)
            
        new_state,reward,done,_ = env.step(action)

        new_state = np.append(new_state, env.get_env()._game.player_vel_y)
        new_state = np.append(new_state, env.get_env()._game.player_rot)

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
            
            states_t = torch.as_tensor(states , dtype = torch.float32 , device = DEVICE)
            dones_t = torch.as_tensor(dones , dtype = torch.int64 , device = DEVICE).unsqueeze(1)
            new_states_t = torch.as_tensor(new_states , dtype = torch.float32 , device = DEVICE)
            actions_t = torch.as_tensor(actions , dtype = torch.int64 , device = DEVICE).unsqueeze(1)
            rewards_t = torch.as_tensor(rewards , dtype=torch.float32 , device = DEVICE).unsqueeze(1)
            
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
        diff_time = time.time() - checkpoint_time
        checkpoint_time = time.time()
        avg = sum(average_list) / len(average_list)
        t = 0
        for i in range(5) :
            t += test(q_network,env)
        t /= 5  
        if t > best_value :
            best_value = t
            best_model.load_state_dict(q_network.state_dict())
            torch.save(best_model.state_dict(), "d3qn_flappy_bird")
            print(f" episode {episode} - timestep {timestep} - test reward (best value): {t} - avg : {avg} - epsilon {epsilon} - diff time {diff_time}s - rb {len(replay_buffer)}")
        else : 
            print(f" episode {episode} - timestep {timestep} - test reward : {t} - avg : {avg} - epsilon {epsilon} - diff time {diff_time}s - rb {len(replay_buffer)}")
        list_tests.append(t)
        summary_writer.add_scalar('Rewards' , t , global_step=episode)
        summary_writer.add_scalar('average_last_rewards' , avg , global_step=episode)


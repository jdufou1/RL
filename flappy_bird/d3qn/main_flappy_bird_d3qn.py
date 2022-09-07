"""
Test file for the D3QN paramteters of the flappy bird environment
"""

import torch
import torch.nn as nn
import numpy as np
import flappy_bird_gym
import time
from tqdm import tqdm

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


env = flappy_bird_gym.make('FlappyBird-v0')
nb_actions = 2
nb_observations = 4

q_network = DuelingQNetwork(nb_actions,nb_observations)
q_network.load_state_dict(torch.load("d3qn_flappy_bird", map_location=torch.device('cpu')))

def play_env(fps=-1):
    state = env.reset()
    state = np.append(state, env._game.player_vel_y)
    state = np.append(state, env._game.player_rot)
    done = False
    cum_sum = 0
    score = 0

    while not done :
        state_t = torch.as_tensor(state , dtype=torch.float32).unsqueeze(0)
        action = torch.argmax(q_network(state_t)).item()
        new_state,reward,done,info = env.step(int(action))
        new_state = np.append(new_state, env._game.player_vel_y)
        new_state = np.append(new_state, env._game.player_rot)
        cum_sum += reward
        score = info["score"]
        if fps>0:
            env.render()
            time.sleep(1/fps)
        state = new_state

    return cum_sum,score


average_score = 0
average_cum_sum = 0
nb_eval = 100
for i in tqdm(range(nb_eval)) :
    cum_sum,score = play_env(fps=-1)
    average_score += score
    average_cum_sum += cum_sum
average_score /= nb_eval
average_cum_sum /= nb_eval 

print(f"score moyen sur {nb_eval}  : {average_score} et moyen des rewards : {average_cum_sum}")
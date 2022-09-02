"""
Test Cartpole
Reinforce
"""
import sys
sys.path.insert(1, '../class/')

import gym
import torch
from torch.autograd import Variable
import time
import numpy as np

from algos.Reinforce import Reinforce
from Env import Env

class EnvCartPole(Env) : 

    def __init__(self,env) -> None:
        super().__init__()
        self.env = env

    def reset(self) : 
        return self.env.reset()

    def step(self,action) : 
        return self.env.step(action)

    def reward_function(self, stp1 , reward ,done , probability):
        return reward

    def transition_function(self, stp1 , reward ,done , probability):
        return probability

class AgentRandom:
    """
         A simple random agent
    """
    def __init__(self,env):
        self.env = env

    def act(self,obs):
        return self.env.action_space.sample()

    def store(self,obs,action,new_obs,reward):
        pass


class Agent :

    def __init__(self,env,model,nb_action):
        self.env = env
        self.model = model
        self.nb_action = nb_action

    def act(self,obs):
        state = torch.from_numpy(obs).float().unsqueeze(0)
        probs = self.model(Variable(state))
        highest_prob_action = np.random.choice(self.nb_action, p=np.squeeze(probs.detach().numpy()))
        return highest_prob_action

    def store(self,obs,action,new_obs,reward):
        pass



def play_env(agent,max_ep=20000000,fps=-1,verbose=False):
    obs = agent.env.reset()
    cumr = 0
    for i in range(max_ep):
        last_obs = obs
        action = agent.act(obs)
        obs,reward,done,info = agent.env.step(int(action))
        agent.store(last_obs,action,obs,reward)
        cumr += reward
        if fps>0:
            agent.env.render()
            if verbose: print(f"iter {i} : {action}: {reward} -> {obs} ")        
            time.sleep(1/fps)
        if done:
            print("fin anticipé")
            break
    return cumr

import os

if __name__ == "__main__":

    env_gym = gym.make('CartPole-v0')

    env_cartpole = EnvCartPole(env_gym)

    PATH = "./weights/reinforce_nn_weights_cartpole"

    algo = Reinforce(nb_observation = 4,nb_action = 2,nb_neurons = 10,env = env_cartpole ,nb_episode = 1000)

    algo.set_model()

    algo.load_weights(PATH)

    algo.set_optim()

    algo.learning(gamma = 0.99,path = PATH)

    agent = Agent(env_gym,model = algo.model , nb_action = 2)

    cum_r = play_env(agent , fps = 30)

    print("recompense totale : ",cum_r)
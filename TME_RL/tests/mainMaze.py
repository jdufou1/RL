import sys
sys.path.insert(1, '../class/')

from Env import Env
from algos.QLearning import QLearning
from AgentPolicy import AgentPolicy

import gym
import gym_maze
import numpy as np
import time

class EnvMaze(Env) : 

    def __init__(self, env) -> None:
        super().__init__()
        self.env = env

    def reset(self):
        res = self.env.reset()
        return tuple(res)

    def step(self, action) :
        stp1 , reward ,done , probability = self.env.step(action)
        return tuple(stp1) , reward ,done , probability

    def reward_function(self, stp1 , reward ,done , probability):
        return reward

    def transition_function(self, stp1 , reward ,done , probability):
        return probability

def play_env(agent,max_ep=500,fps=-1,verbose=False):
    """
        Play an episode :
        * agent : agent with two functions : act(state) -> action, and store(state,action,state,reward)
        * max_ep : maximal length of the episode
        * fps : frame per second,not rendering if <=0
        * verbose : True/False print debug messages
        * return the cumulative reward
    """
    obs = agent.env.reset()
    cumr = 0
    for i in range(max_ep):
        last_obs = obs
        action = agent.act(tuple(obs))
        obs,reward,done,info = agent.env.step(int(action))
        agent.store(last_obs,action,obs,reward)
        cumr += reward
        if fps>0:
            agent.env.render()
            if verbose: print(f"iter {i} : {action}: {reward} -> {obs} ")        
            time.sleep(1/fps)
        if done:
            break
    return cumr

if __name__ == "__main__":

    # Environnement Maze gym
    # credit : https://github.com/MattChanTK/gym-maze

    env_gym = gym.make('maze-random-30x30-plus-v0')

    env = EnvMaze(env_gym)

    size_maze = 100
    S = [(i,j) for i in range(size_maze) for j in range(size_maze)]

    A = dict()
    for s in S :
        A[s] = { 0 : [(None,None,None,None)],1 : [(None,None,None,None)],2 : [(None,None,None,None)],3 : [(None,None,None,None)]}

    algo = QLearning(S,A,env)

    algo.learning()

    policy = algo.get_policy()

    agent = AgentPolicy(env_gym,policy)
    
    cum_r = play_env(agent , fps = 2)

    print("recompense totale : ",cum_r)
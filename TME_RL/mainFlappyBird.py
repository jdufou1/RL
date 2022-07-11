import gym
import flappy_bird_gym
import time
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
#from policy_iteration import *
#from value_iteration import *
#from Qlearning import *

from Env import Env
from QLearning import QLearning
from DoubleQLearning import DoubleQLearning

PROBA = 0
STATE = 1
REWARD = 2
DONE = 3

class MyFlappyEnv:
    """ Custom Flappy Env :
        * state : [horizontal delta of the next pipe, vertical delta, vertical velocity]
    """

    def __init__(self):
        self.env = flappy_bird_gym.make('FlappyBird-v0')
        self.env._normalize_obs = False
        self._last_score = 0
    def __getattr__(self,attr):
        return self.env.__getattribute__(attr)
    
    def step(self,action):
        obs, reward, done, info = self.env.step(action)
        if done:
            reward -=1000
        player_x = self.env._game.player_x
        player_y = self.env._game.player_y

        return np.hstack([obs,self.env._game.player_vel_y]),reward, done, info
    def reset(self):
        return np.hstack([self.env.reset(),self.env._game.player_vel_y])

def test_gym(fps=30):
    env = gym.make('Taxi-v3')
    env.reset()
    r = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        r += reward
        env.render()
        time.sleep(1/fps)
        print(f"iter {i} : action {action}, reward {reward}, state {type(obs)} ")
        if done:
            break
    print(f"reward cumulatif : {r} ")
 
def test_flappy(fps=30):
    env = flappy_bird_gym.make('FlappyBird-v0')
    env.reset()
    r = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        r += reward
        env.render()
        time.sleep(1/fps)
        print(f"iter {i} : action {action}, reward {reward}, state {obs} {info}, {env._game.player_vel_y}")
        print("data : ",env._game)
        if done:
            break
    print(f"reward cumulatif : {r} ")
 
def play_env(agent,max_ep=200000,fps=-1,verbose=False):
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
        action = agent.act(obs)
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

def transform_policy(env,policy) :
    # On transforme notre dictionnaire en Pi : State -> Int (indice action)
    new_policy = dict()
    for state in policy:
        actions = env.env.P[state]
        for action in actions :
            if actions[action] == policy[state]:
                new_policy[state] = action
    return new_policy

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

class AgentPolicy:
    """
        Agent following a policy pi : pi is a dictionary state -> action
    """
    def __init__(self,env,pi):
        self.env = env
        self.pi = pi

    def act(self,obs):
        return self.pi[obs]

    def store(self,obs,action,new_obs,reward):
        pass

    def getPi(self) :
        return self.pi

    def getEnv(self) : 
        return self.env

    def setPi(self,pi):
        self.pi = pi

def arg_state_horizontal(value):
    value = value*100
    step = 185 *2
    min_distance = 25
    max_distance = 165
    tab = np.linspace(min_distance,max_distance,step)
    bestIndex = 0
    index = 0
    while index < len(tab) and value > tab[index]:
        bestIndex = index
        index+=1
    return bestIndex

def arg_state_vertical(value):
    value = value*100
    step = 95 * 2
    min_distance = -30
    max_distance = 65
    tab = np.linspace(min_distance,max_distance,step)
    bestIndex = 0
    index = 0
    while index < len(tab) and value > tab[index]:
        bestIndex = index
        index+=1
    return bestIndex

class FlappyAgent:
    """
        Agent following a policy pi : pi is a dictionary state -> action
    """
    def __init__(self,env,pi):
        self.env = env
        self.pi = pi

    def act(self,obs):
        horizontal_value,vertical_value = obs[0],obs[1]
        current_speed = self.env._game.player_vel_y
        #return self.pi[(arg_state_horizontal(horizontal_value),arg_state_vertical(vertical_value))]
        return self.pi[(round((horizontal_value*100),0),round((vertical_value*100),0),current_speed)]

    def store(self,obs,action,new_obs,reward):
        pass

    def getPi(self) :
        return self.pi

    def getEnv(self) : 
        return self.env

    def setPi(self,pi):
        self.pi = pi

def argmaxQlearning(actions , state, Q) :
    if len(actions) < 1 :
        return None
    else :  
        best_action = actions[0]
        best_value = Q[(state,best_action)]
        for action in actions :
            value = Q[(state,action)]
            if value > best_value :
                best_value = value
                best_action = action
        return best_action

def maxQlearning(actions , state, Q) :
    if len(actions) < 1 :
        return None
    else :  
        best_action = actions[0]
        best_value = Q[(state,best_action)]
        for action in actions :
            value = Q[(state,action)]
            if value > best_value :
                best_value = value
        return best_value


class EnvFlappyBird(Env) : 

    def __init__(self, env) -> None:
        super().__init__()
        self.env = env

    def reset(self):
        horizontal_value,vertical_value = self.env.reset()
        player_vel_y = self.env._game.player_vel_y
        return (round((horizontal_value*100),0),round((vertical_value*100),0),player_vel_y)

    def step(self, action) :
        stp1 , reward ,done , probability = self.env.step(action)
        player_vel_y = self.env._game.player_vel_y
        stp1 = (round((stp1[0]*100),0),round((stp1[1]*100),0),player_vel_y)
        return stp1 , self.reward_function(stp1 , reward ,done , probability) , done , self.transition_function( stp1 , reward ,done , probability)

    def reward_function(self, stp1 , reward ,done , probability):
        if done :
            return -1000
        else :
            return 1

    def transition_function(self, stp1 , reward ,done , probability):
        return probability

if __name__ == "__main__":

    # Environnement FrozenLake de gym

    PATH_MODEL_A = "./q_table_a_flappy_bird.pkl"
    PATH_MODEL_B = "./q_table_b_flappy_bird.pkl"

    env_gym = flappy_bird_gym.make('FlappyBird-v0')

    env = EnvFlappyBird(env_gym)

    S = [(i,j,speed) for i in range(0,167,1) for j in range(-70,70,1) for speed in range(-11,11,1)]

    A = dict()
    for s in S :
        A[s] = { 0 : [(None,None,None,None)],1 : [(None,None,None,None)]}

    algo = DoubleQLearning(S,A,env,PATH_MODEL_A,PATH_MODEL_B)

    algo.learning()

    algo.save_model(PATH_MODEL_A,PATH_MODEL_B)



    policy = algo.get_policy()

    agent = FlappyAgent(env_gym,policy)
    
    cum_r = play_env(agent , fps = 30)

    print("recompense totale : ",cum_r)
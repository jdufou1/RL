import sys
sys.path.insert(1, '../class/')

import flappy_bird_gym
import time
import numpy as np

from Env import Env
from algos.DoubleQLearning import DoubleQLearning

"""
functions
"""

def f(x) : 
    rest = x % 1
    if rest > 0.5 :
        a = rest - 0.5
        if a > 0.25 :
            return np.ceil(x)
        else : 
            return np.floor(x) + 0.5
    else :
        if rest > 0.25 :
            return np.floor(x) + 0.5
        else : 
            return np.floor(x)

def transform_state(horizontal_value , vertical_value , player_vel_y) : 
    return  f(horizontal_value*100) , f(vertical_value*100) , player_vel_y

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

"""
class
"""

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
        return self.pi[transform_state(horizontal_value , vertical_value , current_speed)]

    def store(self,obs,action,new_obs,reward):
        pass

    def getPi(self) :
        return self.pi

    def getEnv(self) : 
        return self.env

    def setPi(self,pi):
        self.pi = pi

class EnvFlappyBird(Env) : 

    def __init__(self, env) -> None:
        super().__init__()
        self.env = env

    def reset(self):
        horizontal_value,vertical_value = self.env.reset()
        player_vel_y = self.env._game.player_vel_y
        return  transform_state(horizontal_value , vertical_value , player_vel_y)# (round((horizontal_value*100),0),round((vertical_value*100),0),player_vel_y)

    def step(self, action) :
        stp1 , reward ,done , probability = self.env.step(action)
        player_vel_y = self.env._game.player_vel_y
        stp1 = transform_state(stp1[0] , stp1[1] , player_vel_y) # (round((stp1[0]*100),0),round((stp1[1]*100),0),player_vel_y)
        return stp1 , self.reward_function(stp1 , reward ,done , probability) , done , self.transition_function( stp1 , reward ,done , probability)

    def reward_function(self, stp1 , reward ,done , probability):
        if done :
            return -1000
        else :
            return 1

    def transition_function(self, stp1 , reward ,done , probability):
        return probability

if __name__ == "__main__":

    # Environnement Flappy Bird de gym

    PATH_MODEL_A = "./weights/q_table_a_flappy_bird.pkl"
    PATH_MODEL_B = "./weights/q_table_b_flappy_bird.pkl"

    env_gym = flappy_bird_gym.make('FlappyBird-v0')

    env = EnvFlappyBird(env_gym)

    S = [(i/2,j/2,speed) for i in range(0,(( 167 + 0 )* 2),1) for j in range(-70 * 2,70 * 2,1) for speed in range(-11,11,1)]

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
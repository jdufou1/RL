import gym
import flappy_bird_gym
import time
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
#from policy_iteration import *
#from value_iteration import *
#from Qlearning import *

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


def arg_state(value):
    value = value
    step = 50
    min_distance = -0.2
    max_distance = 1.8
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
        return self.pi[(arg_state(horizontal_value),arg_state(vertical_value))]

    def store(self,obs,action,new_obs,reward):
        pass

    def getPi(self) :
        return self.pi

    def getEnv(self) : 
        return self.env

    def setPi(self,pi):
        self.pi = pi
















import random
from tqdm import tqdm



    
    

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

def Qlearning(env, gamma = 0.9, eps=0.5, alpha=0.05,nbiter = 100000) :
    step = 50
    """1 - Initialisation of the Q-table """
    Q = dict()
    states = [(i,j) for i in range(step) for j in range(step)]
    for state in states:
        for action in [0,1]:
            Q[(state,action)] = 0 
    actions = [0,1]

    print("Execution Q learning algorithm")
    for _ in tqdm(range(nbiter)) :
        env.reset()
        """ On initialise le premier état """
        horizontal_value,vertical_value = env._get_observation()[0],env._get_observation()[1]
        current_state = (arg_state(horizontal_value),arg_state(vertical_value))
        final_state_reached = False
        """2 - Tant qu'on a pas atteint un état final"""
        iter = 0
        reward_t = 0
        while not final_state_reached :
            """2.1 - On doit choisir une action a faire """
            if random.uniform(0,1) < eps :
                """Exploration : on selectionne une action au hasard"""
                action = actions[random.randint(0, (len(actions)-1))]
            else :
                """Exploitation : on selectionne l'action avec la valeur maximale"""
                action = argmaxQlearning(actions,current_state,Q)
            """2.2 - Mise à jour de la table Q"""
            obs,reward,done,_ = env.step(int(action))
            horizontal_value,vertical_value = obs[0],obs[1]

            nextState = (arg_state(horizontal_value),arg_state(vertical_value))
            
            """2.3 - Verification de l'etat final"""
            final_state_reached = done
            if final_state_reached :
                reward_t = -reward_t

            # print((horizontal_value,vertical_value))
            # print('state : ',nextState,' reward : ',reward_t)

            Q[(current_state,action)] = Q[(current_state,action)] + (alpha * (reward_t + ((gamma * maxQlearning(actions , nextState, Q)) - Q[(current_state,action)])))
            
            """2.4 - Mise à jour de l'état courant"""  
            current_state = nextState
            iter+=1
            reward_t+=1
            if iter > 102 :
                print("yes")
        """3 - transformation de la table Q en politique""" 
        #print("etat final atteint en ",iter,"itérations")
    print(Q)
    PI = dict()
    for state in states:
        #print(state)
        bestAction = actions[0]
        bestValue = Q[(state,bestAction)]
        for action in actions : 
            value = Q[(state,action)]
            if value > bestValue :
                bestValue = value
                bestAction = action
        PI[state] = bestAction
    # print(PI)
    # print(Q.values())
    return PI




















if __name__ == "__main__":

    # Environnement FlappyBird de gym

    envFlappy = flappy_bird_gym.make('FlappyBird-v0')

    #agentRandom = FlappyAgent(envFlappy) 
    #cumr_agentRandom = play_env(agentRandom,verbose=True)
    envFlappy.reset()
    print("donnees du jeu :")
    print(" position x du joueur" , envFlappy._game.player_x)
    print(" position y du joueur" , envFlappy._game.player_y)
    print(" largeur de la carte" , envFlappy._game._screen_width)
    print(" hauteur de la carte" , envFlappy._game._screen_height)
    print("upper_pipes : ",envFlappy._game.upper_pipes)
    print("lower_pipes : ",envFlappy._game.lower_pipes)
    print("_get_observation : ",envFlappy._get_observation())
    policy = Qlearning(envFlappy)



    myAgentQlearning = FlappyAgent(envFlappy,policy)
    cumr_myAgentQlearning = play_env(myAgentQlearning,fps=30)

    print("Recompense cumulé agent Qlearning : ",cumr_myAgentQlearning)
    #test_flappy()
    #print(envFlappy.action_space)

    # play_env(AgentRandom(envFlappy),fps=60)
    
    
from collections import defaultdict
import gym
import flappy_bird_gym
import time
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from policy_iteration import *
from value_iteration import *

PROBA = 0
ETAT = 1
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

# def sum_policy(pi1,pi2):
#     sum = 0
#     for state in pi1 :
#         sum += abs(pi1[state] - pi2[state])
#     return sum





def argmaxQlearning(env , state, Q) :
    if len(env.P[state]) < 1 :
        return None
    else :  
        best_action = [*env.P[state]][0]
        best_value = Q[(state,best_action)]
        for action in [*env.P[state]] :
            value = Q[(state,action)]
            if value > best_value :
                best_value = value
                best_action = action
        return best_action

def maxQlearning(env , state, Q) :
    if len(env.P[state]) < 1 :
        return None
    else :  
        best_action = [*env.P[state]][0]
        best_value = Q[(state,best_action)]
        for action in [*env.P[state]] :
            value = Q[(state,action)]
            if value > best_value :
                best_value = value
        return best_value


def Qlearning(env, gamma , eps=0.5, alpha=0.5) :
    """1 - Initialisation de Q """
    Q = dict()
    for state in env.P :
        #print(state)
        for action in env.P[state] : 
            # print(action)
            Q[(state,action)] = 0
            
    current_state = [*env.P][0] # On choisit le premier état pour commencer

    print("current_state" , current_state)
    # print(env.P[current_state])
    # print([*env.P[current_state]])
    final_state_reached = False
    #print(Q)
    # print(env.P[current_state][random.randint(0, (len(env.P[state])-1))])
    """2 - Tant qu'on a pas atteint un état final"""
    while not final_state_reached :
        #print("iter Qlearning")
        """2.1 - On doit choisir une action a faire """
        if random.uniform(0,1) < eps :
            """Exploration : on selectionne une action au hasard"""
            #print("Exploration")
            action = [*env.P[current_state]][random.randint(0, (len([*env.P[current_state]])-1))]
        else :
            """Exploitation : on selectionne l'action avec la valeur maximale"""
            action = argmaxQlearning(env,current_state,Q)
        """2.2 - Mise à jour de la table Q"""

        #print("Q : ",Q)
        # print("argmaxQlearning(env,current_state,Q) : ",argmaxQlearning(env,current_state,Q))
        # print("Q[(current_state,action)] : ",Q[(current_state,action)])
        
        Q[(current_state,action)] = Q[(current_state,action)] 
        + alpha * (env.P[current_state][action][0][REWARD] 
                    + (gamma * maxQlearning(env , state, Q) 
                    - Q[(current_state,action)]))
        """2.3 - Mise à jour de l'état courant"""           
        current_state = env.P[current_state][action][0][ETAT]
        """2.4 - Verification de l'etat final"""
        print("current_state : ",current_state)
        final_state_reached = env.P[current_state][action][0][DONE] == True
    """3 - transformation de la table Q en politique""" 
    print("etat final atteint")

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

import random

if __name__ == "__main__":

    # Environnement taxi-v3 de gym

    envTaxi = gym.make('Taxi-v3')

    gamma = 0.95
    thresh = 0.001
    
    """ Value Iteration """
    time_valueIter_start = time.time()
    value_iteration(envTaxi , gamma , thresh,itermax=1000)
    time_valueIter_end = time.time()

    """ Policy Iteration """
    time_policyIter_start = time.time()
    policy_iteration(envTaxi , gamma , thresh,itermax=1000)
    time_policyIter_end = time.time()

    """A decommenter pour utiliser l'algorithme de policy iteration ou value iteration """
    best_policy_taxi = transform_policy(envTaxi,policy_iteration(envTaxi , gamma , thresh,itermax=1000))
    # best_policy_taxi = transform_policy(envTaxi,value_iteration(envTaxi , gamma , thresh,itermax=1000))
    
    """Definition des agents"""
    myAgent = AgentPolicy(envTaxi,best_policy_taxi)
    agentRandom = AgentRandom(envTaxi)  

    # Calcul de la somme cumulée = récompense total de la partie
    # fps = 30 normalement
    # cumr_myAgent = play_env(myAgent,fps = 30)
    # cumr_agentRandom = play_env(agentRandom)

    # Affichage     
    # print("Recompense cumulé agent policy : ",cumr_myAgent)
    # print("Recompense cumulé agent random : ",cumr_agentRandom)

    print("--- Temps de convergence de la politique optimale pour l'environemment Taxi ---")
    print("Policy Iteration : ",(time_policyIter_end - time_policyIter_start),"ms")
    print("Value Iteration : ",(time_valueIter_end - time_valueIter_start),"ms")

    Qlearning(envTaxi, gamma , 0.2)

    # envFlappy = flappy_bird_gym.make('FlappyBird-v0')
    # envFlappy.reset()
    # print(envFlappy.action_space)

    # play_env(AgentRandom(envFlappy),fps=60)
    
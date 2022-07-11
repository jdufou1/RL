"""
@author : Jérémy Dufourmantelle
File with necessary functions for the Qlearning algorithm
Environnemment :
State : Id
Action : (Proba,State,Reward,FinalState?)
Stochastic MDP aren't accepted
"""

import random
from tqdm import tqdm

"""
Index values for Actions
"""
PROBA = 0
STATE = 1
REWARD = 2
DONE = 3

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

def Qlearning(env, gamma = 0.6, eps=0.2, alpha=0.05,nbiter = 100000) :
    """1 - Initialisation of the Q-table """
    Q = dict()
    for state in env.P :
        for action in env.P[state] : 
            Q[(state,action)] = random.uniform(0,1)

    for state in env.P :
        for action in env.P[state] : 
            if env.P[state][action][0][DONE] : 
                for action_final_state in env.P[env.P[state][action][0][STATE]] :
                    Q[(env.P[state][action][0][STATE],action_final_state)] = 0
                    
    print("Execution Q learning algorithm")
    for _ in tqdm(range(nbiter)) : 
        current_state = [*env.P][random.randint(0, (len([*env.P])-1))] # On choisit le premier état pour commencer
        final_state_reached = False
        """2 - Tant qu'on a pas atteint un état final"""
        iter = 0
        while not final_state_reached :
            """2.1 - On doit choisir une action a faire """
            if random.uniform(0,1) < eps :
                """Exploration : on selectionne une action au hasard"""
                action = [*env.P[current_state]][random.randint(0, (len([*env.P[current_state]])-1))]
            else :
                """Exploitation : on selectionne l'action avec la valeur maximale"""
                action = argmaxQlearning(env,current_state,Q)
            """2.2 - Mise à jour de la table Q"""
            nextState = env.P[current_state][action][0][STATE]
            Q[(current_state,action)] = Q[(current_state,action)] + (alpha * (env.P[current_state][action][0][REWARD] + ((gamma * maxQlearning(env , nextState, Q)) - Q[(current_state,action)])))
            
            """2.3 - Verification de l'etat final"""
            final_state_reached = env.P[current_state][action][0][DONE]

            """2.4 - Mise à jour de l'état courant"""  
            current_state = nextState
            iter+=1
        """3 - transformation de la table Q en politique""" 
        #print("etat final atteint en ",iter,"itérations")
    PI = dict()
    for state in env.P :
        #print(state)
        bestAction = [*env.P[state]][0]
        bestValue = Q[(state,bestAction)]
        for action in [*env.P[state]] : 
            value = Q[(state,action)]
            if value > bestValue :
                bestValue = value
                bestAction = action
        PI[state] = bestAction
    # print(PI)
    # print(Q.values())
    return PI
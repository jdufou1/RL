"""
@author : Jérémy Dufourmantelle
File with necessary functions for the value iteration algorithm
Environnemment :
State : Id
Action : (Proba,State,Reward,FinalState?)
Stochastic MDP aren't accept
"""

import random

"""
Index values for Actions
"""
PROBA = 0
STATE = 1
REWARD = 2
DONE = 3

"""
Functions
"""

def max_value_iteration(actions , V , gamma) :
    if len(actions) < 1:
        return None
    else :
        first_action_key = [*actions][0]
        first_action = actions[first_action_key][0]
        best_value = first_action[REWARD] + (gamma * V[first_action[STATE]])
        for index_action in [*actions] :
            action = actions[index_action][0]
            value = action[REWARD] + (gamma * V[action[STATE]])
            if value > best_value :
                best_value = value
        return best_value

def argmax_value_iteration(actions , V , gamma) :
    if len(actions) < 1:
        return None
    else :
        first_action_key = [*actions][0]
        best_action = actions[first_action_key][0]
        best_value = best_action[REWARD] + (gamma * V[best_action[STATE]])
        for index_action in [*actions] :
            action = actions[index_action][0]
            value = action[REWARD] + (gamma * V[action[STATE]])
            if value > best_value :
                best_action = action
                best_value = value
        return best_action

def calcul_diff_btw_two_V(env , V1 , V2) :
    result = 0
    for state in env.P :
        result += abs(V1[state] - V2[state])
    return result

def value_iteration(env , gamma , thresh, itermax) : 
    """ Initialisation """
    V = dict()
    for state in env.P :
        V[state] = 0
    """ Boucler jusqu'a convergence """
    delta = 1e10
    i = 0
    while delta > thresh and i < itermax :
        delta = 0
        # print("iteration value iteration : ",i)
        for s in env.P : 
            v = V[s]
            """ Mise a jour de la valeur de l'état s """
            V[s] = max_value_iteration(env.P[s] , V , gamma)
            """ Calcul de la difference d'amelioration """
            delta = max(delta,abs(v - V[s]))
        i += 1
    print("Nombre d'itération value iteration : ",i)
    """ Recuperation de la politique optimale """
    pi = dict()
    for s in env.P :
        pi[s] = [argmax_value_iteration(env.P[s] , V , gamma)]
    return pi
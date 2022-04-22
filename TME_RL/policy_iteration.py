"""
@author : Jérémy Dufourmantelle
File with necessary functions for the policy iteration algorithm
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

def test_convergence(policy1 , policy2):
    """
    policy1 : dict(state : [(proba,état,reward,done)])
    policy2 : dict(state : [(proba,état,reward,done)])
    Return True if the policy 1 is the same than policy 2
    """
    for state in policy1 :
        if policy1[state] != policy2[state] :
            return False
    return True

def eval_pol(pi,env,gamma,thresh):
    V_pi = dict()
    for state in pi :
        V_pi[state] = 0
    newV_pi = dict()
    for state in pi :
        newV_pi[state] = 1e10
    iter = 0
    delta = 1e10
    while delta > thresh:
        delta = 0
        newV_pi = V_pi.copy()
        for state in pi :
            if not pi[state][0][DONE] : # if the State isn't final
                V_pi[state] = pi[state][0][REWARD] + (gamma * V_pi[pi[state][0][STATE]])
                delta = max(delta , abs(V_pi[state] - newV_pi[state]))
        iter += 1
    return V_pi

def get_pol(V , env , gamma):
    pi = dict()
    for etat in V :
        actions = env.P[etat]
        bestAction = actions[0]
        bestValue = bestAction[0][REWARD] + gamma * V[bestAction[0][STATE]]
        for action in env.P[etat] :
            value_action = actions[action][0][REWARD] + gamma * V[actions[action][0][STATE]]
            if value_action > bestValue :
                bestValue = value_action
                bestAction = actions[action]
        pi[etat] = bestAction
    return pi

def policy_iteration(env , gamma , thresh,itermax):
    """ Initialisation of V """
    V = dict()
    for state in env.P :
        V[state] = 0
    """ Initialisation of the first policy """
    PI = dict()
    for state in env.P :
        PI[state] = env.P[state][random.randint(0, (len(env.P[state])-1))]
    convergence = False
    while not convergence:
        """ Evaluation of the current policy"""
        V = eval_pol(PI,env,gamma,thresh)
        """ Improvement of the current policy """
        PI2 = PI.copy()
        PI = get_pol(V , env , gamma)
        """ Convergence test """
        convergence = test_convergence(PI,PI2)
    return PI
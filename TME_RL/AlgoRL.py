from Env import Env

import random

import torch

class AlgoRL :

    def __init__(self) -> None:
        """
        S : list of states
        A : dict of actions (states : {action_1 : [(proba,state,reward,done)_1 , ... , (proba,state,reward,done)_k], ... , action_n : [(...) , ... , (...)]})
        In model-free : proba,reward,done are None
        """
        self.pi = dict()

    def learning(self):
        pass

    def get_policy(self):
        return self.pi

class AlgoRLMB(AlgoRL) : 
    """
    abstract class for the model-based RL algorithms
    """
    def __init__(self, S: list, A: dict) -> None:
        super().__init__()
        self.S = S
        self.A = A
        self.V = dict()
        self.initialisation_policy()
        self.initialisation_v_function()
        

    def initialisation_policy(self):
        for s in self.S :
            self.pi[s] = list(self.A[s].keys())[random.randint(0, (len(list(self.A[s].keys()))-1))]

    def initialisation_v_function(self):
        for s in self.S :
            self.V[s] = 0
    
class AlgoRLMF(AlgoRL) :
    """
    abstract class for the model-free RL algorithms
    """
    def __init__(self, S: list, A: dict, env : Env) -> None:
        super().__init__()
        self.S = S
        self.A = A
        self.Q = dict()
        self.env = env
        self.initialisation_policy()
        self.initialisation_q_function()

    def initialisation_policy(self):
        for s in self.S :
            self.pi[s] = list(self.A[s].keys())[random.randint(0, (len(list(self.A[s].keys()))-1))]

    def initialisation_q_function(self):
        for s in self.S :
            for a in list(self.A[s].keys()) :
                self.Q[(s,a)] = 0

    def choose_action(self , decision, st , eps):
        if decision == "e-greedy" : 
            if random.uniform(0,1) < eps :
                at = list(self.A[st].keys())[random.randint(0, (len(list(self.A[st].keys()))-1))]
            else :
                at = self.argmaxQL(st)
        elif decision == "random" :
            at = self.A[st][random.randint(0, (len(self.A[st])-1))]
        elif decision == "greedy" :
            at = self.argmaxQL(st)
        elif decision == "proportionnelle" :
            at = None
            pass
        elif decision == "softmax" :
            at = None
        else :
            at = None
        return at

    def argmaxQL(self, state) :
        assert len(self.A[state]) >= 1 , "current state haven't action"
        best_action = list(self.A[state].keys())[0]
        best_value = self.Q[(state,best_action)]
        for action in list(self.A[state].keys()) :
            value = self.Q[(state,action)]
            if value > best_value :
                best_value = value
                best_action = action
        return best_action

class AlgoRLPB(AlgoRL) :
    """
    abstract class for the model-free policy-based RL algorithms
    """
    def __init__(self,nb_observation : int ,nb_action : int,nb_neurons : int, env : Env) -> None:
        super().__init__()
        self.nb_observation = nb_observation
        self.nb_action = nb_action
        self.nb_neurons = nb_neurons
        self.env = env
    
    def set_model(self) :
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.nb_observation,self.nb_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(self.nb_neurons,self.nb_action),
            torch.nn.Softmax()
        )

    def set_optim(self,lr=0.01) :
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=lr)
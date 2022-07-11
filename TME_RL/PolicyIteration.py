from AlgoRL import AlgoRLMB

import random
import numpy as np


class PolicyIteration(AlgoRLMB) :

    def __init__(self, S: list, A: dict) -> None:
        super().__init__(S, A)
        

    def eval_pol(self,gamma,thresh):
        delta = thresh + 1e-10
        while delta > thresh:
            delta = 0
            for s in self.S :
                v = self.V[s]
                self.V[s] = np.array([proba * (reward + (gamma * self.V[state])) for (proba,state,reward,_) in self.A[s][self.pi[s]]]).sum()
                delta = max(delta , abs(v - self.V[s]))

    def get_pol(self, gamma):
        policy_stable = True
        for s in self.S :
            old_action = self.pi[s]
            best_action = list(self.A[s].keys())[0]
            best_value = np.array([proba * (reward + (gamma * self.V[state])) for (proba,state,reward,_) in self.A[s][best_action]]).sum()
            for a in list(self.A[s].keys()) :
                value = np.array([proba * (reward + (gamma * self.V[state])) for (proba,state,reward,_) in self.A[s][a]]).sum()
                if value > best_value :
                    best_action = a
                    best_value = value
            self.pi[s] = best_action
            if old_action != self.pi[s] :
                policy_stable = False
        return policy_stable

    def learning(self , gamma = 0.99, thresh = 1e-10,itermax = 1000):
        policy_stable = False
        iteration = 0
        while not policy_stable and iteration < itermax:
            self.eval_pol(gamma,thresh)
            policy_stable = self.get_pol(gamma)
            iteration += 1
        print(f"Apprentissage realisé en {iteration} itérations")
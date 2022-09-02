from AlgoRL import AlgoRLMB

import random
import numpy as np

class ValueIteration(AlgoRLMB) :

    def __init__(self, S: list, A: dict) -> None:
        super().__init__(S, A)

    def learning(self , gamma = 0.99, thresh = 1e-10,itermax = 1000):
        delta = thresh + 1e-10
        iteration = 0
        while delta > thresh and iteration < itermax:
            delta = 0
            for s in self.S :
                v = self.V[s]
                self.V[s] = np.array([np.array([proba * (reward + (gamma * self.V[state])) for (proba,state,reward,_) in self.A[s][a] ]).sum() for a in list(self.A[s].keys())]).max()
                delta = max(delta , abs(v - self.V[s]))
            iteration += 1
        self.update_policy(gamma)
        print(f"Apprentissage realisé en {iteration} itérations")

    def update_policy(self,gamma) :
        for s in self.S :
            best_action = list(self.A[s].keys())[0]
            best_value = np.array([proba * (reward + (gamma * self.V[state])) for (proba,state,reward,_) in self.A[s][best_action]]).sum()
            for a in list(self.A[s].keys()) :
                value = np.array([proba * (reward + (gamma * self.V[state])) for (proba,state,reward,_) in self.A[s][a]]).sum()
                if value > best_value :
                    best_action = a
                    best_value = value
            self.pi[s] = best_action
from AlgoRL import AlgoRLMF
from Env import Env

import numpy as np
from tqdm import tqdm

class QLearning(AlgoRLMF) :
    """
    Qlearning , model-free algorithm
    """

    def __init__(self, S: list, A: dict, env: Env) -> None:
        super().__init__(S, A, env)

    def learning(self,df = 0.99 , lr = 0.8, itermax = 5000,decision = "e-greedy",eps = 0.2) :
        self.stats(df,lr,itermax,eps)
        list_reward = list()
        for _ in tqdm(range(itermax)) :
            st = self.env.reset()
            final_state_reached = False
            cum_r = 0
            while not final_state_reached :
                at = self.choose_action(decision , st, eps = eps )
                stp1,reward,done,prob = self.env.step(at)
                final_state_reached = done
                self.Q[(st,at)] =  self.Q[(st,at)] + (lr * (reward + ((df * np.array([self.Q[(stp1 , a)] for a in list(self.A[stp1].keys())]).max()) - self.Q[(st,at)])))
                st = stp1
                cum_r += reward
            list_reward.append(cum_r)
        self.update_policy()
        return list_reward

    def update_policy(self) :
        for s in self.S :
            best_action = list(self.A[s].keys())[0]
            best_value = self.Q[(s,best_action)]
            for a in list(self.A[s].keys()) :
                value = self.Q[(s,a)]
                if value > best_value :
                    best_action = a
                    best_value = value
            self.pi[s] = best_action

    def stats(self,df = 0.99 , lr = 0.8, itermax = 5000,eps = 0.1) : 
        print("Q-LEARNING")
        print("model free and off policy algorithm")
        print("-------------------------")
        print("eps : ",eps)
        print("discount factor : ",df)
        print("learning rate : ",lr)
        print("iteration max : ",itermax)
        print("-------------------------")
            
from AlgoRL import AlgoRLMF
from Env import Env

import numpy as np
from tqdm import tqdm

class MC(AlgoRLMF) :
    """
    Monte-carlo method , model-free algorithm
    """

    def __init__(self, S: list, A: dict, env: Env) -> None:
        super().__init__(S, A, env)
        self.reward_returns = dict()
        for s in self.S :
            for a in list(self.A[s].keys()) :
                self.reward_returns[(s,a)] = list()

    def learning(self, itermax = 15000, eps = 0.01, gamma = 0.99) :
        for _ in tqdm(range(itermax)) :
            st = self.env.reset()

            at = self.choose_action("e-greedy" , st, eps = eps )
            stp1,reward,done,_ = self.env.step(at)
            episode = [(st,at,reward)]
            st = stp1
            final_state_reached = done

            while not final_state_reached :
                
                at = self.choose_action("e-greedy" , st, eps = eps )
                stp1,reward,done,_ = self.env.step(at)
                episode.append((st,at,reward))
                st = stp1
                final_state_reached = done

            G = 0
            for (s,a,r) in reversed(episode) :
                G = (G * gamma) + r
                self.reward_returns[(s,a)].append(G)
                self.Q[(s,a)] = np.array(self.reward_returns[(s,a)]).mean()

        self.update_policy()

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
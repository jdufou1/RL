from AlgoRL import AlgoRLMF
from Env import Env

from tqdm import tqdm

class Sarsa(AlgoRLMF) :
    """
    Sarsa , model-free algorithm
    """

    def __init__(self, S: list, A: dict, env: Env) -> None:
        super().__init__(S, A, env)

    def learning(self,df = 0.99 , lr = 0.8, itermax = 100000,decision = "e-greedy",eps = 0.1) :
        list_reward = list()
        for _ in tqdm(range(itermax)) :
            st = self.env.reset()
            final_state_reached = False
            cum_r = 0
            at = self.choose_action(decision , st, eps = eps )
            while not final_state_reached :
                stp1,reward,done,_ = self.env.step(at)
                atp1 = self.choose_action(decision , stp1, eps = eps )
                final_state_reached = done
                self.Q[(st,at)] =  self.Q[(st,at)] + (lr * (reward + ((df * self.Q[(stp1 , atp1)]) - self.Q[(st,at)])))
                st = stp1
                at = atp1
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
            
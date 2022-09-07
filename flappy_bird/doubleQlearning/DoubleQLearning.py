from AlgoRL import AlgoRLMF
from Env import Env

import numpy as np
import random
from tqdm import tqdm

import pickle
import time


class DoubleQLearning(AlgoRLMF) :
    """
    DoubleQLearning , model-free algorithm
    paper : https://proceedings.neurips.cc/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf
    """

    STEP_SAVE_TEST = 1000

    def __init__(self, S: list, A: dict, env: Env, path_table_a = None, path_table_b = None) -> None:
        super().__init__(S, A, env)

        self.save = path_table_a is not None and path_table_b is not None

        if self.save:
            print("[DoubleQLearning] : model loaded")
            file_a = open(path_table_a , "rb")
            self.Q_a = pickle.load(file_a)
            file_a.close()
            file_b = open(path_table_b , "rb")
            self.Q_b = pickle.load(file_b)
            file_b.close()
            self.path_table_a = path_table_a
            self.path_table_b = path_table_b
        else : 
            self.Q_a = dict()
            self.Q_b = dict()
            for s in self.S :
                for a in list(self.A[s].keys()) :
                    self.Q_a[(s,a)] = 0
                    self.Q_b[(s,a)] = 0

    def learning(self,df = 0.99 , lr = 0.8, itermax = 10,eps = 0.1) :
        self.stats(df,lr,itermax,eps)
        list_reward = list()
        for i in range(itermax) :
            print(i)
            st = self.env.reset()
            final_state_reached = False
            cum_r = 0
            while not final_state_reached :
                at = self.choose_action(st,eps)

                stp1,reward,done,prob = self.env.step(at)
                final_state_reached = done
                if random.uniform(0,1) < 0.5 : 
                    # update A
                    self.Q_a[(st,at)] =  self.Q_a[(st,at)] + (lr * (reward + ((df * np.array([self.Q_b[(stp1 , a)] for a in list(self.A[stp1].keys())]).max()) - self.Q_a[(st,at)])))
                else : 
                    # update B
                    self.Q_b[(st,at)] =  self.Q_b[(st,at)] + (lr * (reward + ((df * np.array([self.Q_a[(stp1 , a)] for a in list(self.A[stp1].keys())]).max()) - self.Q_b[(st,at)])))
                st = stp1
                cum_r += reward
        
            list_reward.append(cum_r)
            if self.save and i % DoubleQLearning.STEP_SAVE_TEST == 0 :
                self.save_model(self.path_table_a , self.path_table_b)
                # test : 
                cum_rew = 0.0
                st = self.env.reset()
                done = False
                start_time = time.time()
                while not done :
                    action = self.argmaxDQL(st)
                    stp1,reward,done,_ = self.env.step(action)
                    st = stp1
                    cum_rew += reward
                end_time = time.time()
                print(f"episode : {i}/{itermax} - test reward {cum_rew} - time : {(end_time - start_time)} s")



        self.update_policy()
        return list_reward


    def choose_action(self, st, eps):
        if random.uniform(0,1) < eps :
            return list(self.A[st].keys())[random.randint(0, (len(list(self.A[st].keys()))-1))]
        else :
            return self.argmaxDQL(st)
        

    def argmaxDQL(self, state) :
        assert len(self.A[state]) >= 1 , "current state haven't action"
        best_action = list(self.A[state].keys())[0]
        best_value = self.Q_a[(state,best_action)] + self.Q_b[(state,best_action)]
        for action in list(self.A[state].keys()) :
            value = self.Q_a[(state,action)] + self.Q_b[(state,action)]
            if value > best_value :
                best_value = value
                best_action = action
        return best_action

    def update_policy(self) :
        for s in self.S :
            best_action = list(self.A[s].keys())[0]
            best_value = self.Q_a[(s,best_action)] + self.Q_b[(s,best_action)]
            for a in list(self.A[s].keys()) :
                value = self.Q_a[(s,a)] + self.Q_b[(s,a)]
                if value > best_value :
                    best_action = a
                    best_value = value
            self.pi[s] = best_action


    def save_model(self,path_table_a , path_table_b) : 

        # os.remove(path_table_a)
        # os.remove(path_table_b)

        print("[DoubleQLearning] : model saved")
        file_a = open(path_table_a, "wb")
        pickle.dump(self.Q_a, file_a)
        file_a.close()

        file_b = open(path_table_b, "wb")
        pickle.dump(self.Q_b, file_b)
        file_b.close()

    def stats(self,df = 0.99 , lr = 0.8, itermax = 5000,eps = 0.1) : 
        print("DOUBLE Q-LEARNING")
        print("model free and off policy algorithm")
        print("-------------------------")
        print("eps : ",eps)
        print("discount factor : ",df)
        print("learning rate : ",lr)
        print("iteration max : ",itermax)
        print("-------------------------")
            
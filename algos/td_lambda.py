"""
https://amreis.github.io/ml/reinf-learn/2017/11/02/reinforcement-learning-eligibility-traces.html
"""

from tqdm import tqdm
import numpy as np
import random
import gym




from torch.utils.tensorboard import SummaryWriter



class TD_lambda :

    def __init__(self, 
                env,
                lambda_value = 0.95, 
                learning_rate = 0.8, 
                discount_factor = 0.99, 
                epsilon = 0.1,
                test_frequency = 10
                ) -> None:
        self.env = env
        self.lambda_value = lambda_value
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.test_frequency = test_frequency
        self.action_value_function = np.zeros((self.env.observation_space.n , self.env.action_space.n))

    def get_action(self,state,epsilon):
        if random.random() < epsilon :
            return self.env.action_space.sample()
        else : 
            return np.argmax( np.array( [self.action_value_function[state,action] for action in range(self.env.action_space.n)] ) )

    def train(self,summary_writer, batchsize = 10000) : 
        i = 0
        
        for i in tqdm(range(batchsize)) : 
            eligibility_trace = np.zeros((self.env.observation_space.n , self.env.action_space.n))
            
            done = False
            state = self.env.reset()
            while not done and i < batchsize : 
                action = self.get_action(state,self.epsilon)
                new_state, reward, done, _ = self.env.step(action)
                
                eligibility_trace[state,action] *= self.lambda_value * self.discount_factor
                eligibility_trace[state,action] += 1

                target = reward + self.discount_factor * np.max(np.array( [self.action_value_function[new_state,action] for action in range(self.env.action_space.n)]))
                td_error = target - self.action_value_function[state,action]
                self.action_value_function[state,action] += self.learning_rate * td_error * eligibility_trace[state,action]
                
                state = new_state
            
            # test
            if i % self.test_frequency == 0 :
                cum_reward = 0.0
                state = self.env.reset()
                done = False
                while not done : 
                    action = self.get_action(state,0)
                    new_state, reward, done, _ = self.env.step(action)
                    cum_reward += reward
                    state = new_state
                #print(f"Test : iteration {i}/{batchsize} - reward : {cum_reward}")
                summary_writer.add_scalar('Reward' , reward , global_step=i)


            

# test


log_dirs = ['./log/taxiv3_td_lambda_01',
'./log/taxiv3_td_lambda_02',
'./log/taxiv3_td_lambda_03',
'./log/taxiv3_td_lambda_05',
'./log/taxiv3_td_lambda_07',
'./log/taxiv3_td_lambda_08',
'./log/taxiv3_td_lambda_09',
'./log/taxiv3_td_lambda_95']

lambda_values = [
    0.1,0.2,0.3,0.5,0.7,0.8,0.9,0.95
]

env = gym.make("Taxi-v3")
print("self.env.action_space.n : ",env.action_space.n)
print("env.observation_space : ",env.observation_space.n)

for log_dir,lambda_value in zip(log_dirs,lambda_values) :
    summary_writer = SummaryWriter(log_dir)
    algo = TD_lambda(env,lambda_value=lambda_value)
    algo.train(summary_writer)
import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3")

nb_observations = env.observation_space.n
nb_actions = env.action_space.n

action_value_function = np.zeros((nb_observations,nb_actions))

nb_iteration = 20000
discount_factor = 0.99
learning_rate = 5e-2
test_frequency = 10
eps = 0.15

def take_action_eps_greedy(state,eps) -> int :
    if random.random() > eps :
        return np.argmax(action_value_function[state,:])
    else: 
        return np.random.choice(np.arange(nb_actions))

def softmax(actions) : 
    return np.exp(actions) / np.exp(actions).sum()

def test() :
    state = env.reset()
    done = False
    cum_sum = 0
    while not done :
        action = take_action_eps_greedy(state,0)
        new_state,reward,done,_ = env.step(action)
        cum_sum += reward
        state = new_state
    return cum_sum

list_test = list()

for i in range(nb_iteration):

    state = env.reset()
    done = False
    while not done : 
        action = take_action_eps_greedy(state,eps)
        new_state,reward,done,_ = env.step(action)
        action_value_function[state , action] += learning_rate * (
                                                        reward + 
                                                        discount_factor * 
                                                        (1 - done) *
                                                        (softmax(action_value_function[new_state , :]) * action_value_function[new_state , :]).sum() - 
                                                        action_value_function[state , action]
                                                    )
    
        state = new_state

    if i % test_frequency == 0 :
        cum_sum_test = test()
        list_test.append(cum_sum_test)
        print(f"iteration {i} - test reward : {cum_sum_test}")


plt.figure()
plt.title("Expected Sarsa - Taxi-v3 - rewards")
plt.xlabel("iteration")
plt.ylabel("rewards")
plt.plot(np.arange(0,nb_iteration,test_frequency),list_test,label="rewards")
plt.legend()
plt.show()
                            
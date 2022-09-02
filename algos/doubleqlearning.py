import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3")

nb_observations = env.observation_space.n
nb_actions = env.action_space.n

action_value_function_a = np.zeros((nb_observations,nb_actions))
action_value_function_b = np.zeros((nb_observations,nb_actions))

nb_iteration = 20000
discount_factor = 0.99
learning_rate = 0.05
test_frequency = 100
eps = 0.1

def take_action_eps_greedy(state,eps) -> int :
    if random.random() > eps :
        return np.argmax(action_value_function_a[state,:] + action_value_function_b[state,:])
    else: 
        return np.random.choice(np.arange(nb_actions))

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


for i in range(nb_iteration) : 

    state = env.reset()
    done = False
    while not done :
        
        action = take_action_eps_greedy(state,eps)
        new_state,reward,done,_ = env.step(action)
        if random.random() < 0.5 :
            # update A
            action_a = np.argmax(action_value_function_a[new_state,:])

            action_value_function_a[state,action] += learning_rate * (
                reward + 
                discount_factor *
                (1 - done) * 
                action_value_function_b[new_state,action_a] -
                action_value_function_a[state,action]
            )
        else :
            # update B
            action_b = np.argmax(action_value_function_b[new_state,:])

            action_value_function_b[state,action] += learning_rate * (
                reward + 
                discount_factor *
                (1 - done) * 
                action_value_function_a[new_state,action_b] -
                action_value_function_b[state,action]
            )
        state = new_state

    if i % test_frequency == 0 :
        cum_sum_test = test()
        list_test.append(cum_sum_test)
        print(f"iteration {i} - test reward : {cum_sum_test}")


plt.figure()
plt.title("QLEARNING - Taxi-v3 - rewards")
plt.xlabel("iteration")
plt.ylabel("rewards")
plt.plot(np.arange(0,nb_iteration,test_frequency),list_test,label="rewards")
plt.legend()
plt.show()
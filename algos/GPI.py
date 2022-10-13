import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


env = gym.make("Taxi-v3")

nb_actions = env.action_space.n
nb_observation = env.observation_space.n

discount_factor = 0.99

print("nb_actions : ",nb_actions)
print("nb_observation : ",nb_observation)




# policy evaluation

def policy_evaluation(
    state_value_function,
    policy,
    max_iter = 5
    ) : 
    delta = 1
    terminaison_criteria = 1e-3
    iteration = 0
    while iteration < max_iter and delta > terminaison_criteria :
        delta = 0
        for state in range(nb_observation) :
            v = state_value_function[state]
            best_action = policy[state]
            data = env.env.P[state][best_action]
            probas = np.asarray([i[0] for i in data])
            rewards = np.asarray([i[2] for i in data])
            value_new_states = np.asarray([state_value_function[i[1]] for i in data])
            dones = np.asarray([i[3] for i in data])
            state_value_function[state] = (probas * (rewards + discount_factor * (1-dones) * value_new_states)).sum()
            delta = max(delta , abs(v - state_value_function[state]))
        iteration += 1
    return state_value_function,policy,iteration

# policy improvement : 
def policy_improvement(
    state_value_function,
    policy
) :
    policy_stable = True
    for state in range(nb_observation) :
        old_action = policy[state]

        best_action = 0
        data = env.env.P[state][best_action]
        probas = np.asarray([i[0] for i in data])
        rewards = np.asarray([i[2] for i in data])
        value_new_states = np.asarray([state_value_function[i[1]] for i in data])
        dones = np.asarray([i[3] for i in data])
        best_value_action = (probas * (rewards + discount_factor * (1-dones) * value_new_states)).sum()
        for action in range(nb_actions) : 
            data = env.env.P[state][action]
            probas = np.asarray([i[0] for i in data])
            rewards = np.asarray([i[2] for i in data])
            value_new_states = np.asarray([state_value_function[i[1]] for i in data])
            dones = np.asarray([i[3] for i in data])
            value_action = (probas * (rewards + discount_factor * (1-dones) * value_new_states)).sum()
            if value_action > best_value_action :
                best_value_action = value_action
                best_action = action

        policy[state] = best_action
        if old_action != policy[state] :
            policy_stable = False

    return policy_stable


def generalized_policy_iteration(
    state_value_function,
    policy,
    max_iter = 5
) :
    nb_iter_val = 0
    iteration = 0
    done = False
    while not done :
        # print(f"iteration {iteration}")
        state_value_function,policy,nb_iter = policy_evaluation(state_value_function,policy,max_iter)
        nb_iter_val += nb_iter
        done = policy_improvement(state_value_function,policy)
        iteration += 1
    return nb_iter_val

def test_gpi(
    nb_eval_list : list,
) :
    list_time = list()
    list_nb_iter = list()
    for nb_eval in tqdm(nb_eval_list) :

        # initialisation
        state_value_function = np.zeros(nb_observation)
        policy = np.zeros(nb_observation)
        
        start_time = time.time()
        nb_iter_val = generalized_policy_iteration(state_value_function,policy,nb_eval)
        end_time = time.time()
    
        tot_time = end_time - start_time
        print(tot_time)
        list_time.append(tot_time)
        list_nb_iter.append(nb_iter_val)
    
    return list_time,list_nb_iter

nb_eval_list = np.arange(0,50,1)

list_time,list_nb_iter = test_gpi(nb_eval_list)



plt.figure()
plt.title("Temps en fonction du nombre de d'itération max - GPI - Taxi-v3")
plt.xlabel("MaxValue")
plt.ylabel("time (s)")
plt.plot(list_time)
plt.show()

plt.figure()
plt.title("Nombre d'itération en fonction du nombre de d'itération max - GPI - Taxi-v3")
plt.xlabel("MaxValue")
plt.ylabel("Nombre d'itération")
plt.plot(list_nb_iter)
plt.show()

# print(f"test reward : {cum_sum}")

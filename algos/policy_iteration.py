import gym
import numpy as np
import matplotlib.pyplot as plt


env = gym.make("Taxi-v3")

nb_actions = env.action_space.n
nb_observation = env.observation_space.n

discount_factor = 0.99

print("nb_actions : ",nb_actions)
print("nb_observation : ",nb_observation)

# initialisation
state_value_function = np.zeros(nb_observation)
policy = np.zeros(nb_observation)

# policy evaluation

def policy_evaluation() : 
    delta = 1
    terminaison_criteria = 1e-3
    iteration = 0
    while iteration < 1 and delta > terminaison_criteria :
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

# policy improvement : 
def policy_improvement() :
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

def policy_iteration() :
    iteration = 0
    done = False
    while not done :
        print(f"iteration {iteration}")
        policy_evaluation()
        done = policy_improvement()
        iteration += 1

policy_iteration()

# Test
state = env.reset()
done = False
cum_sum = 0
while not done : 
    new_state,reward,done,_ = env.step(policy[state])
    cum_sum += reward
    state = new_state

print(f"test reward : {cum_sum}")
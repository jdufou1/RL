import gym
import numpy as np

env = gym.make("Taxi-v3")

nb_actions = env.action_space.n
nb_observations = env.observation_space.n

state_value_function = np.zeros(nb_observations)
policy = np.zeros(nb_observations)

discount_factor = 0.99

theta = 1e-6

delta = 1

iteration = 0

while delta > theta :
    print(f"iteration {iteration} - delta : {delta}") 
    delta = 0
    for state in range(nb_observations) :
        v = state_value_function[state]
        data = env.env.P[state][0]
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
    
        state_value_function[state] = best_value_action
        delta = max(delta , abs(v - state_value_function[state]))
    iteration += 1

for state in range(nb_observations) :
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

# Test
state = env.reset()
done = False
cum_sum = 0
while not done : 
    new_state,reward,done,_ = env.step(policy[state])
    cum_sum += reward
    state = new_state

print(f"test reward : {cum_sum}")
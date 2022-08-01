
from collections import namedtuple

import gym
import torch
from torch.distributions import Categorical
from torch.optim import Adam
import torch.nn as nn

env = gym.make('CartPole-v0')

obs_size = env.observation_space.shape[0]
num_actions = env.action_space.n

Rollout = namedtuple('Rollout',
                     ['states', 'actions', 'rewards', 'next_states', ])


def train(epochs=1000, num_rollouts=10):
    for epoch in range(epochs):
        rollouts = []

        for t in range(num_rollouts):
            state = env.reset()
            done = False

            samples = []

            while not done:
                with torch.no_grad():
                    action = get_action(state)
                    next_state, reward, done, _ = env.step(action)

                    # Collect samples
                    samples.append((state, action, reward, next_state))

                    state = next_state

            # Transpose our samples
            states, actions, rewards, next_states = zip(*samples)

            states = torch.stack([torch.from_numpy(state) for state in states], dim=0).float()
            next_states = torch.stack([torch.from_numpy(state) for state in next_states], dim=0).float()
            actions = torch.as_tensor(actions).unsqueeze(1)
            rewards = torch.as_tensor(rewards).unsqueeze(1)

            rollouts.append(Rollout(states, actions, rewards, next_states))

        update_agent(rollouts)

        # test 

        if epoch % 5 == 0 :
            cum_sum = 0.0
            obs = env.reset()
            done = False
            while not done:
                action = get_action(obs)
                new_obs, rew, done, _ = env.step(action)
                obs = new_obs
                cum_sum += rew
                
            print(f"episode - {epoch} - reward test : {cum_sum}")




actor_hidden = 32
actor = nn.Sequential(nn.Linear(4, actor_hidden),
                      nn.ReLU(),
                      nn.Linear(actor_hidden, num_actions),
                      nn.Softmax(dim=1))

def get_action(state):
    state = torch.tensor(state).float().unsqueeze(0)  # Turn state into a batch with a single element
    #print(actor(state))
    dist = Categorical(actor(state))  # Create a distribution from probabilities for actions
    return dist.sample().item()


# Critic takes a state and returns its values
critic_hidden = 32
critic = nn.Sequential(nn.Linear(4, critic_hidden),
                       nn.ReLU(),
                       nn.Linear(critic_hidden, 1))

critic_optimizer = Adam(critic.parameters(), lr=0.005)

def update_critic(advantages):
    loss = .5 * (advantages ** 2).mean()
    critic_optimizer.zero_grad()
    loss.backward()
    critic_optimizer.step()

def update_agent(rollouts):
    states = torch.cat([r.states for r in rollouts], dim=0)
    actions = torch.cat([r.actions for r in rollouts], dim=0).flatten()

    advantages = [estimate_advantages(states, next_states[-1], rewards) for states, _, rewards, next_states in rollouts]
    advantages = torch.cat(advantages, dim=0).flatten()

    update_critic(advantages)

    distribution = actor(states)
    distribution = torch.distributions.utils.clamp_probs(distribution)
    probabilities = distribution[range(distribution.shape[0]), actions]

    # We will calculate the gradient wrt to the new probabilities (surrogate function),
    # so second probabilities should be treated as a constant
    L = surrogate_loss(probabilities, probabilities.detach(), advantages)
    KL = kl_div(distribution, distribution)
    parameters = list(actor.parameters())

    g = flat_grad(L, parameters, retain_graph=True)  # We will use the graph several times
    d_kl = flat_grad(KL, parameters, create_graph=True)  # Create graph, because we will call backward() on it (for HVP)


    def HVP(v):
        return flat_grad(d_kl @ v, parameters, retain_graph=True)

    search_dir = conjugate_gradient(HVP, g, max_iterations=10)

    delta = 0.01 # Should be low (approximately betwween 0.01 and 0.05
    
    max_length = torch.sqrt(2 * delta / (search_dir @ HVP(search_dir)))
    max_step = max_length * search_dir


    def criterion(step):
        # Apply parameters' update
        apply_update(step)

        with torch.no_grad():
            distribution_new = actor(states)
            distribution_new = torch.distributions.utils.clamp_probs(distribution_new)

            probabilities_new = distribution_new[range(distribution_new.shape[0]), actions]

            L_new = surrogate_loss(probabilities_new, probabilities, advantages)
            KL_new = kl_div(distribution, distribution_new)

        L_improvement = L_new - L
        if L_improvement > 0 and KL_new <= delta:
            return True

        # Step size too big, reverse
        apply_update(-step)
        return False

    i = 0
    while not criterion((0.9 ** i) * max_step) and i < 10:
        i += 1

def estimate_advantages(states, last_state, rewards):
    values = critic(states)
    last_value = critic(last_state.unsqueeze(0))
    
    next_values = torch.zeros_like(rewards)
    for i in reversed(range(rewards.shape[0])):
        last_value = next_values[i] = rewards[i] + 0.99 * last_value
        
    advantages = next_values - values
    return advantages

def surrogate_loss(new_probabilities, old_probabilities, advantages):
    return (new_probabilities / old_probabilities * advantages).mean()

def kl_div(p, q):
    p = p.detach()
    return (p * (p.log() - q.log())).sum(-1).mean()

def flat_grad(y, x, retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True

    g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    g = torch.cat([t.view(-1) for t in g])
    return g

def conjugate_gradient(A, b, delta=0., max_iterations=float('inf')):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()

    i = 0
    while i < max_iterations:
        AVP = A(p)
        dot_old = r @ r
        alpha = dot_old / (p @ AVP)

        x_new = x + alpha * p

        if (x - x_new).norm() <= delta:
            return x_new

        i += 1
        r = r - alpha * AVP

        beta = (r @ r) / dot_old
        p = r + beta * p

        x = x_new
    return x


def apply_update(grad_flattened):
    n = 0
    for p in actor.parameters():
        numel = p.numel()
        g = grad_flattened[n:n + numel].view(p.shape)
        p.data += g
        n += numel


train()
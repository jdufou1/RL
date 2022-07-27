"""
Trust Region Policy Optimization
source/algo : https://dac.lip6.fr/wp-content/uploads/2021/10/cours-3.pdf
paper : https://arxiv.org/pdf/1502.05477.pdf
help : https://github.com/joschu/modular_rl/blob/master/modular_rl/trpo.py
"""

"""
Import 
"""
from torch.utils.tensorboard import SummaryWriter
import gym
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

"""
Hyper parameters
"""
LOG_DIR = './log/trpo_cartpole'
KL_DIVERGENCE_LIMIT = 0.01
BACKTRACKING_COEFF = 0.1
BACKTRACKING_STEPS = 10
CONJUGATE_GRADIENT_STEPS = 10
CONJUGATE_GRADIENT_RATIO = 0.1
NUMBER_EPISODES = 2048
TEST_TIME_STEP = 5
LR_VALUE_FUNCTION = 0.01
BATCH_SIZE = 128
DISCOUNT_FACTOR = 0.99

"""
Functions
"""

def conjugate_gradient_algorithm(H_hat_k , g_hat_k) :
    x = torch.zeros(g_hat_k.shape)
    r_k = g_hat_k - (H_hat_k @ x)
    d = r_k
    for _ in range(CONJUGATE_GRADIENT_STEPS) :
        alpha_k = (torch.transpose(r_k , 1 , 0) @ r_k) / ((torch.transpose(d , 1 , 0) @ H_hat_k) @ d)
        
        x = x + ( alpha_k * d ) 
        r_kp1 = r_k - ( alpha_k * H_hat_k @ d )
        if r_kp1[r_kp1 <= CONJUGATE_GRADIENT_RATIO ].all(): 
            break
        beta = (torch.transpose(r_kp1 , 1 , 0) @ r_kp1 )/ (torch.transpose(r_k , 1 , 0) @ r_k )
        d = r_kp1 + ( beta * d )

        r_k = r_kp1

    return x

def kullback_leibler(theta , theta_k , F) :
    return 0.5 * torch.transpose((theta - theta_k), 1 , 0) @ F @ (theta - theta_k)

def line_search_algorithm(x_k , module , F, states_t, nn, A_t_t):
    """
    F : fisher matrix
    """
    nn.zero_grad()

    with torch.no_grad():
        theta_k = torch.nn.utils.parameters_to_vector(module.weight.data.clone()).unsqueeze(1)
        params_theta_k = module.weight.data.clone()



    for j in range(BACKTRACKING_STEPS) :

        delta = np.sqrt((2 * KL_DIVERGENCE_LIMIT) / torch.transpose(x_k , 1 , 0) @ F @ x_k) * x_k

        theta = theta_k + ( BACKTRACKING_COEFF**j * delta )

        # calcul de la distance de kullback leibler

        kl_value = kullback_leibler(theta, theta_k , F)

        # calcul des logs proba suivant la nouvelle politique
        
        theta = theta.squeeze()

        with torch.no_grad():
            params = torch.reshape(theta , module.weight.shape)
            module.weight = torch.nn.Parameter(params)

        _, log_probas_t = nn.act(Variable(states_t))
        
        (log_probas_t * A_t_t).mean().backward()

        with torch.no_grad():
            log_gradient_policy = torch.nn.utils.parameters_to_vector(module.weight.grad).unsqueeze(1)



        L_theta_k_of_theta = ( ( 1.0 / (1.0  - DISCOUNT_FACTOR) ) * ( torch.transpose(log_gradient_policy , 1 , 0) ) @ (theta.unsqueeze(1) - theta_k) ) 

        if kl_value <= KL_DIVERGENCE_LIMIT and L_theta_k_of_theta >= 0 :
            module.weight = torch.nn.Parameter(params)
            break
        else :
            module.weight = torch.nn.Parameter(params_theta_k)
        nn.zero_grad()

"""
Neural networks
"""

class Network_Value_Function(nn.Module) : 

    def __init__(self,env) -> None:
        super().__init__()
        self.env = env
        self.nb_obses = int(np.prod(env.observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(self.nb_obses , 64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self,x) :
        return self.net(x)

class Network_Policy_Function(nn.Module) : 

    def __init__(self,env,linear1,linear2) -> None:
        super().__init__()
        self.env = env
        self.nb_actions = env.action_space.n
        self.nb_obses = int(np.prod(env.observation_space.shape))

        self.net = nn.Sequential(
            linear1,
            nn.ReLU(),
            linear2
        )

    def forward(self,x) :
        return self.net(x)

    def act(self,obs) :
        obs_t = torch.as_tensor(obs , dtype=torch.float32)
        probs = self.net(obs_t)
        if obs_t.shape[0] == 4 :
            probs = probs.unsqueeze(0)
        list_prob = torch.stack([probs[i][torch.argmax(probs,dim = 1)[i]] for i in range(probs.shape[0])])
        log_prob = torch.log(list_prob)
        return [torch.argmax(probs,dim = 1)[i].detach().item() for i in range(probs.shape[0])], log_prob

summary_writer = SummaryWriter(LOG_DIR)
env = gym.make('CartPole-v0')

value_function = Network_Value_Function(env)

linear1 = nn.Linear(int(np.prod(env.observation_space.shape)) , 64)
linear2 = nn.Linear(64,env.action_space.n)
modules = [linear1,linear2]
policy_function = Network_Policy_Function(env,linear1,linear2)

optimizer_value_function = torch.optim.Adam(value_function.parameters(), lr=LR_VALUE_FUNCTION)

for k in range(NUMBER_EPISODES):

    D_k = list()
    R_t = list()
    A_t = list()
    trajectory = list()

    # 1. Rollouts
    obs = env.reset()
    for i in range(BATCH_SIZE):
        action,log_proba = policy_function.act(obs)
        new_obs, rew, done, _ = env.step(action[0])
        transition = (obs, action, rew, done, new_obs,log_proba)
        D_k.append(transition)
        trajectory.append((rew,obs,new_obs))

        obs = new_obs
        if done:
            obs = env.reset()
            # enregistrement des rewards accumulées : 
            for t in range(len(trajectory)) : 
                Gt = 0
                pw = 0 
                for rew,obs,new_obs in trajectory[t:]:      
                    Gt += DISCOUNT_FACTOR**pw * rew
                    pw += 1
                R_t.append(Gt)
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs , dtype = torch.float32).unsqueeze(0)
                    new_obs_t = torch.as_tensor(new_obs , dtype = torch.float32).unsqueeze(0)
                    A_t.append((rew + ( DISCOUNT_FACTOR * value_function(new_obs_t)[0].detach().item()) - value_function(obs_t)[0].detach().item()))
            trajectory = list()
        elif i == BATCH_SIZE - 1 :
            for t in range(len(trajectory)) : 
                Gt = 0
                pw = 0  
                for rew,obs,new_obs in trajectory[t:]:      
                    Gt += DISCOUNT_FACTOR**pw * rew
                    pw += 1
                R_t.append(Gt)
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs , dtype = torch.float32).unsqueeze(0)
                    new_obs_t = torch.as_tensor(new_obs , dtype = torch.float32).unsqueeze(0)
                    A_t.append((rew + ( DISCOUNT_FACTOR * value_function(new_obs_t)[0].detach().item()) - value_function(obs_t)[0].detach().item()))
            

    states = np.asarray([t[0] for t in D_k])
    rews = np.asarray([t[2] for t in D_k])
    dones = np.asarray([t[3] for t in D_k])
    new_states = np.asarray([t[4] for t in D_k])
    R_t_a = np.asarray(R_t)
    A_t_a = np.asarray(A_t)


    states_t = torch.as_tensor(states, dtype = torch.float32)
    rews_t = torch.as_tensor(rews, dtype = torch.float32).unsqueeze(-1)
    dones_t = torch.as_tensor(dones, dtype = torch.float32).unsqueeze(-1)
    new_states_t = torch.as_tensor(new_states, dtype = torch.float32)
    
    R_t_t = torch.as_tensor(R_t_a , dtype = torch.float32)
    A_t_t = torch.as_tensor(A_t_a , dtype = torch.float32)
    
    log_probas = [t[5] for t in D_k]

    log_probas_t = torch.stack(log_probas)
    (log_probas_t).mean().backward()

    for module in modules :

        # Estimate policy gradient
        with torch.no_grad():
            log_gradient_policy = torch.nn.utils.parameters_to_vector(module.weight.grad.data.clone()).unsqueeze(1)
            log_gradient_policy_advantage = log_gradient_policy * A_t_t.sum() 
        
        g_hat_k = log_gradient_policy_advantage / BATCH_SIZE
        
        # Compute the fischer matrix
        H_hat_k = log_gradient_policy @ log_gradient_policy.T

        # 4. Conjugate gradient algorithm
        x_k = conjugate_gradient_algorithm(H_hat_k , g_hat_k)
        
        # 5. Line seach algorithm
        line_search_algorithm(x_k , module , H_hat_k, states_t, policy_function, A_t_t)


    # 6. gradient descent algorithm

    # Compute the loss
    mse = torch.nn.MSELoss(reduction = 'mean')
    loss = mse( R_t_t ,value_function(states_t).squeeze())

    # gradient descent
    optimizer_value_function.zero_grad()
    loss.backward()
    optimizer_value_function.step()

    # Test
    if k % TEST_TIME_STEP == 0 :
        cum_sum = 0.0
        obs = env.reset()
        done = False
        while not done:
            action,_ = policy_function.act(obs)
            new_obs, rew, done, _ = env.step(action[0])
            obs = new_obs
            cum_sum += rew
            
        print(f"episode - {k} - reward test : {cum_sum}")
        summary_writer.add_scalar('Rewards',cum_sum,global_step = k)
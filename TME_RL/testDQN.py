"""
-----------
Flappy Bird - Deep Q-Learning with experience replay 
-----------
"""


"""
Librairies
"""

from collections import deque
import flappy_bird_gym
import time
import numpy as np
import random

import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from Env import Env

"""
Hyper parameters
"""


BATCH_SIZE = 64
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
REPLAY_MEMORY = 50000 # 50000
M = 100000 # number of episode
LEARNING_RATE = 2e-5 # learning_rate
DISCOUNT_FACTOR = 0.99 # discount factor
C = 1000 # step to reinitialise weight
BATCH = 32
OBSERVE = 10000
EXPLORE = 3000000 # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
TRAIN = True
LOG_DIR = './log/DeepQLearningExperienceReplay'

"""
Functions
"""

def play_env(agent,max_ep=5000000,fps=-1,verbose=False):
    """
        Play an episode :
        * agent : agent with two functions : act(state) -> action, and store(state,action,state,reward)
        * max_ep : maximal length of the episode
        * fps : frame per second,not rendering if <=0
        * verbose : True/False print debug messages
        * return the cumulative reward
    """
    obs = agent.env.reset()
    cumr = 0
    for i in range(max_ep):
        last_obs = obs
        action = agent.act(obs)
        obs,reward,done,info = agent.env.step(int(action))
        agent.store(last_obs,action,obs,reward)
        cumr += reward
        if fps>0:
            agent.env.render()
            if verbose: print(f"iter {i} : {action}: {reward} -> {obs} ")        
            time.sleep(1/fps)
        if done:
            break
    return cumr

class FlappyAgent:
    """
        Agent following a policy pi : pi is a dictionary state -> action
    """
    def __init__(self,env,Q):
        self.env = env
        self.Q = Q

    def act(self,obs):
        state_conv = cv2.cvtColor(cv2.resize(obs, (80, 80)), cv2.COLOR_BGR2GRAY)
        state_conv_4 = np.stack((state_conv, state_conv, state_conv, state_conv), axis=2)
        state_t = torch.as_tensor(state_conv_4, dtype = torch.float32, device = DEVICE).unsqueeze(0)
        state_t = torch.transpose(state_t , 1 , 3)
        return self.Q(state_t).argmax().item()

    def store(self,obs,action,new_obs,reward):
        pass

class EnvFlappyBird(Env) : 

    def __init__(self, env) -> None:
        super().__init__()
        self.env = env

    def reset(self):
        return self.env.reset()

    def step(self, action) :
        stp1 , reward ,done , probability = self.env.step(action)
        return stp1 , self.reward_function(stp1 , reward ,done , probability) , done , self.transition_function( stp1 , reward ,done , probability)

    def reward_function(self, stp1 , reward ,done , probability):
        if done :
            return -1000
        else :
            return 1

    def transition_function(self, stp1 , reward ,done , probability):
        return probability

def choose_action(Q, state, eps, nb_action = 2) : 
    if random.uniform(0,1) < eps :
        return random.randint(0, nb_action-1),float(1.0/float(nb_action))
    else: 
        state_t = torch.as_tensor(state , dtype = torch.float32, device = DEVICE).unsqueeze(0)
        state_t = torch.transpose(state_t , 1 , 3)
        return Q(state_t).argmax().item(),Q(state_t).max().item()

if __name__ == "__main__":

    summary_writer = SummaryWriter(LOG_DIR)

    env_gym = flappy_bird_gym.make('FlappyBird-rgb-v0')

    env_gym.reset()

    env = EnvFlappyBird(env_gym)

    replay_memory = deque()

    # First Q : action-value function

    model_q =  nn.Sequential(
            nn.Conv2d(4,32,8,stride = 4,padding=(0,0)),
            nn.MaxPool2d(kernel_size=2 , padding=(0,0)),
            nn.ReLU(),
            nn.Conv2d(32,64,4,stride = 2 , padding = (1,1)),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2, stride = 2,padding = (1,1)),
            nn.ReLU(),
            nn.Conv2d(64,64,3,stride = 1, padding = (1,1)),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2, padding = (1,1)),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.LogSoftmax()
            )

    # model_q.load_state_dict(torch.load("weigths_q"))

    if TRAIN :

        model_q_bis = nn.Sequential(
            nn.Conv2d(4,32,8,stride = 4,padding=(0,0)),
            nn.MaxPool2d(kernel_size=2 , padding=(0,0)),
            nn.ReLU(),
            nn.Conv2d(32,64,4,stride = 2 , padding = (1,1)),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2, stride = 2,padding = (1,1)),
            nn.ReLU(),
            nn.Conv2d(64,64,3,stride = 1, padding = (1,1)),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2, padding = (1,1)),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.LogSoftmax()
            )

        
        # model_q_bis.load_state_dict(torch.load("weigths_q"))

        print("[Model loaded] - Initialisation")

        optimizer = optim.Adam(model_q.parameters(), lr=LEARNING_RATE)

        iteration = 1

        

        for episode in range(1,M):

            reward_counter = 0.0

            state = env.reset()

            state_conv = cv2.cvtColor(cv2.resize(state, (80, 80)), cv2.COLOR_BGR2GRAY)
            state_conv_4 = np.stack((state_conv, state_conv, state_conv, state_conv), axis=2)

            final_state_reached = False

            epsilon = INITIAL_EPSILON

            while not final_state_reached :
                
                
                at,p_at = choose_action(model_q, state_conv_4, eps = epsilon)

                # scale down epsilon
                if epsilon > FINAL_EPSILON and iteration > OBSERVE:
                    epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

                stp1,reward,done,prob = env.step(at)

                reward_counter += reward

                state_conv_ = cv2.cvtColor(cv2.resize(stp1, (80, 80)), cv2.COLOR_BGR2GRAY)
                state_conv_ = state_conv_.reshape(state_conv_.shape[0],state_conv_.shape[1],1 )
                stp1 = np.append(state_conv_, state_conv_4[:, :, :3], axis = 2)

                final_state_reached = done

                transition = (state_conv_4,reward,done,stp1,at)

                replay_memory.append(transition)

                if len(replay_memory) > REPLAY_MEMORY:
                    replay_memory.popleft()

                print(len(replay_memory))

                if iteration > OBSERVE:

                    transitions = random.sample(replay_memory , BATCH_SIZE)

                    states = np.asarray([t[0] for t in transitions])
                    rews = np.asarray([t[1] for t in transitions])
                    dones = np.asarray([t[2] for t in transitions])
                    new_states = np.asarray([t[3] for t in transitions])

                    states_t = torch.as_tensor(states , dtype = torch.float32, device = DEVICE)
                    rews_t = torch.as_tensor(rews , dtype = torch.float32 , device = DEVICE).unsqueeze(-1)
                    dones_t = torch.as_tensor(dones , dtype = torch.float32, device = DEVICE).unsqueeze(-1)
                    new_states_t = torch.as_tensor(new_states , dtype = torch.float32, device = DEVICE)

                    states_t = torch.transpose(states_t , 3 , 1)
                    new_states_t = torch.transpose(new_states_t , 3 , 1)
                    
                    y = rews_t + ( DISCOUNT_FACTOR * dones_t * torch.max(model_q_bis(new_states_t),dim=1)[0] )
                    
                    output = torch.max((model_q(states_t)),dim=1)[0]

                    mse = nn.MSELoss()

                    optimizer.zero_grad()
                    loss = mse(output , y)
                    loss.backward()
                    optimizer.step()

                    state_conv_4 = stp1

                if iteration % C == 0 :
                    print(f"[Model saved] episode : {episode} - iteration {iteration}")
                    torch.save(model_q.state_dict(), "weigths_q")
                    model_q_bis.load_state_dict(torch.load("weigths_q"))
                    
                iteration += 1
                print("iteration : ",iteration,"best action : ",at," max value : ",p_at)

            reward_counter += random.randint(0,100) + 1000

            print("reward_counter ! ",reward_counter)
            summary_writer.add_scalar('Rewards',reward_counter,global_step = episode)

    env.reset()

    agent = FlappyAgent(env_gym,model_q)
    
    cum_r = play_env(agent , fps = 30)

    print("recompense totale : ",cum_r)
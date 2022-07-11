from collections import deque
import gym
import flappy_bird_gym
import time
import numpy as np
import matplotlib.pyplot as plt



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


from Env import Env
from QLearning import QLearning
from DoubleQLearning import DoubleQLearning
 
def play_env(agent,max_ep=500,fps=-1,verbose=False):
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
        
        #return self.pi[(arg_state_horizontal(horizontal_value),arg_state_vertical(vertical_value))]
        
        state = torch.transpose(torch.Tensor(obs), 0 , 2 ).unsqueeze(dim=0)
        # print("shape " ,state.shape)
        #print("results : ",Q(state))
        return self.Q(torch.Tensor(state)).argmax().numpy()


    def store(self,obs,action,new_obs,reward):
        pass

    def getPi(self) :
        return self.pi

    def getEnv(self) : 
        return self.env

    def setPi(self,pi):
        self.pi = pi


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

import random



def choose_action(Q, state, eps, nb_action = 2) : 
    if random.uniform(0,1) < eps :
        return random.randint(0, 1)
    else: 

        # print("predict : ",Q.predict([state],verbose = 0))
        # print("argmax : ",Q.predict([state],verbose = 0).argmax())
        # print("state : ",torch.Tensor(state).shape)
        state = torch.transpose(torch.Tensor(state), 0 , 2 ).unsqueeze(dim=0)
        # print("shape " ,state.shape)
        #print("results : ",Q(state))
        return Q(torch.Tensor(state)).argmax().numpy()


def create_mini_batches(sequence, batch_size = 15):
    # print("seqeunce : ",sequence)
    sequence = np.array(sequence)
    # mini_batches = []
    np.random.shuffle(sequence)
    # n_minibatches = sequence.shape[0] // batch_size
    # i = 0
    # for i in range(n_minibatches + 1):
    #     mini_batch = sequence[i * batch_size:(i + 1)*batch_size]
    #     X_mini = mini_batch[:-1]
    #     mini_batches.append(X_mini)
    # if sequence.shape[0] % batch_size != 0:
    #     mini_batch = sequence[i * batch_size:sequence.shape[0]]
    #     X_mini = mini_batch[:, :-1]
    #     mini_batches.append(X_mini)
    mini_batches =  [sequence[i:i+batch_size] for i in range(0, len(sequence), batch_size)]
    return mini_batches[random.randint(0, len(mini_batches)-1)]
    # mini_batches = np.array(mini_batches)

    # print("mini_batch : ",mini_batches)
    # print(mini_batches[0])
    # print(mini_batches.shape)
    # print(random.randint(0, len(mini_batches)-1))
    # return mini_batches[random.randint(0, len(mini_batches)-1)]

def getYvalue(minibatch,Qb,gamma) : 
    y = list()


    for st,at,r,stp1 in minibatch :
        if stp1 is None :
            y.append(r)
        else:
            stp1 = torch.transpose(torch.Tensor(stp1), 0 , 2 ).unsqueeze(dim=0)
            # print("STP1 : ",stp1.shape)
            y.append( ( r + gamma * (Qb(torch.Tensor(stp1)).detach().numpy()).max() ) )
    return np.array(y)

def getQvalue(minibatch,Q) : 
    X = list()
    for (st,at,r,stp1) in minibatch :
        X.append(  st  )
    return np.array(X)


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


from torch.autograd import Variable

if __name__ == "__main__":


    env_gym = flappy_bird_gym.make('FlappyBird-rgb-v0')

    print("---------------------")

    env_gym.reset()
    # obs,reward,done,info = env_gym.step(1)
    # print(obs.shape)

    print(f"Action space: {env_gym.action_space}")
    print(f"Observation space: {env_gym.observation_space}")

    env = EnvFlappyBird(env_gym)

    replay_memory = deque()

    REPLAY_MEMORY = 400 # 50000

    M = 3000 # number of episode

    lr = 0.8 # learning_rate

    df = 0.99 # discount factor

    C = 100 # step to reinitialise weight

    BATCH = 100

    OBSERVE = 200

    EXPLORE = 2000 # frames over which to anneal epsilon
    FINAL_EPSILON = 0.0001 # final value of epsilon
    INITIAL_EPSILON = 0.1 # starting value of epsilon

    TRAIN = False

    # First Q : action-value function

    model_q =  nn.Sequential(
            nn.Conv2d(3,8,2),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(8,16,3),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(16,20,4),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(4400, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.LogSoftmax()
            )

    model_q.load_state_dict(torch.load("weigths_q"))

    if TRAIN :

        

        # model_q_bis = nn.Sequential(
        #       nn.Conv2d(3,10,5),
        #       nn.MaxPool2d(kernel_size=2),
        #       nn.ReLU(),
        #       nn.Conv2d(10,20,5),
        #       nn.Dropout2d(),
        #       nn.MaxPool2d(kernel_size=2),
        #       nn.ReLU(),
        #       nn.Flatten(),
        #       nn.Linear(172500, 50), #8625
        #       nn.ReLU(),
        #       nn.Dropout2d(),
        #       nn.Linear(50,2),
        #       nn.LogSoftmax()
        #     )
        model_q_bis = nn.Sequential(
            nn.Conv2d(3,8,2),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(8,16,3),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(16,20,4),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(4400, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.LogSoftmax()
            )

        
        model_q_bis.load_state_dict(torch.load("weigths_q"))

        print("[Model loaded] - Initialisation")

        optimizer = optim.SGD(model_q.parameters(), lr=0.01,
                        momentum=0.5)

        iteration = 1

        for episode in range(1,M):

            st = env.reset()

            final_state_reached = False

            epsilon = INITIAL_EPSILON

            while not final_state_reached :
                
                at = choose_action(model_q, st, eps = epsilon)

                # scale down epsilon
                if epsilon > FINAL_EPSILON and iteration > OBSERVE:
                    epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

                stp1,reward,done,prob = env.step(at)
                
                final_state_reached = done
                
                if not final_state_reached :
                    replay_memory.append(np.array([np.array(st), int(at) , 1 , stp1]))
                else :
                    replay_memory.append(np.array([np.array(st), int(at) , -1000 , None]))

                if len(replay_memory) > REPLAY_MEMORY:
                    replay_memory.popleft()

                if iteration > OBSERVE:
                    
                    tmp = np.array(list(replay_memory))

                    examples = iterate_minibatches(tmp, tmp, BATCH)

                    minibatch,_= next(examples)

                    y = getYvalue(minibatch,model_q_bis,df )
                    
                    X = getQvalue(minibatch,model_q)

                    X =  torch.transpose(torch.Tensor(X), 1 , 3 )
    
                    criterion = nn.MSELoss()
                    optimizer.zero_grad()
                    output,_ = torch.max((model_q(torch.Tensor(X))),dim=1)
                    output = output.type(torch.LongTensor)
                    y = torch.Tensor(y).type(torch.FloatTensor)
                    output  = torch.Tensor(output).type(torch.FloatTensor)
                    
                    #loss = F.mse_loss(output, y,reduction='sum') # nll_loss
                    loss = criterion(output , y)
                    loss = Variable(loss, requires_grad = True)
                    loss.backward()
                    optimizer.step()
                    
                    st = stp1

                if iteration % C == 0 :
                    print(f"[Model saved] episode : {episode} - iteration {iteration}")
                    torch.save(model_q.state_dict(), "weigths_q")
                    model_q_bis.load_state_dict(torch.load("weigths_q"))
                    
                iteration += 1

    env.reset()

    agent = FlappyAgent(env_gym,model_q)
    
    cum_r = play_env(agent , fps = 30)

    print("recompense totale : ",cum_r)


   
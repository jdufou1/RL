import gym
import flappy_bird_gym
import time
import numpy as np

from tqdm import tqdm
#from policy_iteration import *
#from value_iteration import *
#from Qlearning import *

from Env import Env

import torch

from torch.autograd import Variable

 
def play_env(agent,max_ep=200000,fps=-1,verbose=False):
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


class EnvFlappyBird(Env) : 

    def __init__(self, env) -> None:
        super().__init__()
        self.env = env

    def reset(self):
        horizontal_value,vertical_value = self.env.reset()
        player_vel_y = self.env._game.player_vel_y
        return np.array([horizontal_value,vertical_value ,player_vel_y])

    def step(self, action) :
        stp1 , reward ,done , probability = self.env.step(action)
        player_vel_y = self.env._game.player_vel_y
        stp1 = np.array([stp1[0],stp1[1],player_vel_y])
        return stp1 , self.reward_function(stp1 , reward ,done , probability) , done , self.transition_function( stp1 , reward ,done , probability)

    def reward_function(self, stp1 , reward ,done , probability):
        if done :
            return -1000
        else :
            return 1

    def transition_function(self, stp1 , reward ,done , probability):
        return probability



class Agent :

    def __init__(self,env,model,nb_action):
        self.env = env
        self.model = model
        self.nb_action = nb_action

    def act(self,obs):

        player_vel_y = self.env._game.player_vel_y
        state = np.array([obs[0],obs[1],player_vel_y])

        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.model(Variable(state))
        highest_prob_action = np.random.choice(self.nb_action, p=np.squeeze(probs.detach().numpy()))
        return highest_prob_action

    def store(self,obs,action,new_obs,reward):
        pass



from Reinforce import Reinforce

if __name__ == "__main__":

    # Environnement FrozenLake de gym

    PATH = "reinforce_nn_weights_flappyBird"

    env_gym = flappy_bird_gym.make('FlappyBird-v0')

    env = EnvFlappyBird(env_gym)

    algo = Reinforce(nb_observation = 3,nb_action = 2,nb_neurons = 100,env = env ,nb_episode = 10000)

    algo.set_model()

    algo.load_weights(PATH)

    algo.set_optim()

    algo.learning(gamma = 0.99,path = PATH)

    agent = Agent(env_gym, algo.model ,2)
    
    cum_r = play_env(agent , fps = 30)

    print("recompense totale : ",cum_r)

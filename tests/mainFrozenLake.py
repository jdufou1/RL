import sys
sys.path.insert(1, '../class/')


from algos.PolicyIteration import PolicyIteration
from algos.ValueIteration import ValueIteration
from algos.QLearning import QLearning
from algos.DoubleQLearning import DoubleQLearning
from algos.Sarsa import Sarsa
from algos.MC import MC
from Env import Env
from AgentPolicy import AgentPolicy

import gym
import time

class EnvFrozenLake(Env) : 

    def __init__(self, env) -> None:
        super().__init__()
        self.env = env

    def reset(self):
        return self.env.reset()

    def step(self, action) :
        res = self.env.step(action)
        #print(res)
        return res

    def reward_function(self, stp1 , reward ,done , probability):
        return reward

    def transition_function(self, stp1 , reward ,done , probability):
        return probability

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

if __name__ == "__main__":

    # Environnement FrozenLake de gym

    env_gym = gym.make('FrozenLake-v0')

    env = EnvFrozenLake(env_gym)

    S = list(env_gym.P.keys())

    A = env_gym.P

    algo = Sarsa(S,A,env)

    algo.learning()

    policy = algo.get_policy()

    agent = AgentPolicy(env_gym,policy)
    
    cum_r = play_env(agent , fps = 30)

    print("recompense totale : ",cum_r)

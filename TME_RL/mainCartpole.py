from random import Random
import gym

import time

class AgentRandom:
    """
         A simple random agent
    """
    def __init__(self,env):
        self.env = env

    def act(self,obs):
        return self.env.action_space.sample()

    def store(self,obs,action,new_obs,reward):
        pass

def play_env(agent,max_ep=200000,fps=-1,verbose=False):
    obs = agent.env.reset()
    cumr = 0
    for i in range(max_ep):
        last_obs = obs
        action = agent.act(obs)
        obs,reward,done,info = agent.env.step(int(action))
        print(obs)
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

    env = gym.make('CartPole-v0')

    agent_random = AgentRandom(env)

    cum_r = play_env(agent_random , fps = 30)

    print("recompense totale : ",cum_r)
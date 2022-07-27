from AlgoRL import AlgoRLPB

from Env import Env

import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

LOG_DIR = './log/reinforce'

class Reinforce(AlgoRLPB) :
    """
    RL algorithm policy-based, model-free 
    Monte carlo sampling
    """

    SAVE_CONSTANT = 1000

    def __init__(self, nb_observation: int, nb_action: int, nb_neurons: int,
     env: Env, nb_episode : int) -> None:

        super().__init__(nb_observation, nb_action, nb_neurons, env)
        self.nb_episode = nb_episode


    def load_weights(self,path) :
        self.model.load_state_dict(torch.load(path))
        print("[Reinforce] : model loaded")


    def get_action(self,state) :
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.model(Variable(state))
        highest_prob_action = np.random.choice(self.nb_action, p=np.squeeze(probs.detach().numpy()))
        # print(probs)
        # print(probs.detach().numpy())
        # print(np.random.choice(self.nb_action, p=np.squeeze(probs.detach().numpy())))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob

    def learning(self, gamma : float , path : str) : 

        summary_writer = SummaryWriter(LOG_DIR)

        episode_count = 0

        for episode in tqdm(range(self.nb_episode)):

            episode_count += 1

            list_reward = list()
            list_log_proba = list()
            state = self.env.reset()
            done = False
            
            while not done :
                action,log_proba = self.get_action(state)
                state,r_,done,i_ = self.env.step(action)
                list_reward.append(r_)
                list_log_proba.append(log_proba)

            discounted_rewards = list()


            if len(list_reward) > 101 :
                print("LEN : ",len(list_reward))

            for t in range(len(list_reward)) : 
                Gt = 0
                pw = 0
                for reward in list_reward[t:]:  
                    Gt += gamma**pw * reward
                    pw += 1
                discounted_rewards.append(Gt)

            discounted_rewards = np.array(discounted_rewards)
            
            # Apprentissage

            discounted_rewards = torch.tensor(discounted_rewards,dtype=torch.float32)
            # version baseline
            discounted_rewards = (discounted_rewards - torch.mean(discounted_rewards)) / (torch.std(discounted_rewards))
            
            log_prob = torch.stack(list_log_proba)
            policy_gradient = -log_prob*discounted_rewards

            self.model.zero_grad()
            policy_gradient.sum().backward()
            self.optimizer.step()

            if episode % Reinforce.SAVE_CONSTANT == 0 :
                torch.save(self.model.state_dict(), path)
                print("[Reinforce] : model saved")

            # logging
            summary_writer.add_scalar('Episodes' , episode_count , global_step=episode)
                
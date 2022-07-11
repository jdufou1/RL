from AlgoRL import AlgoRLPB

from Env import Env

class Reinforce(AlgoRLPB) :

    SAVE_CONSTANT = 1000

    def __init__(self, nb_observation: int, nb_action: int, nb_neurons: int,
     env: Env, nb_episode : int, gamma : float) -> None:

        super().__init__(nb_observation, nb_action, nb_neurons, env)
        self.nb_episode = nb_episode

    def learning(self) : 
        for episode in range(self.nb_episode):
            done = False

            state = self.env.reset()

            action = ... # model()
            state,r_,done,i_ = self.env.step(action)
            while not done :
                action = ... # model()
                state,r_,done,i_ = self.env.step(action)

            if episode % Reinforce.SAVE_CONSTANT == 0 :
                # save_values
                pass
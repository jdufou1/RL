class AgentPolicy:
    """
        Agent following a policy pi : pi is a dictionary state -> action
    """
    def __init__(self,env,pi):
        self.env = env
        self.pi = pi

    def act(self,obs):
        return self.pi[obs]

    def store(self,obs,action,new_obs,reward):
        pass

    def getPi(self) :
        return self.pi

    def getEnv(self) : 
        return self.env

    def setPi(self,pi):
        self.pi = pi
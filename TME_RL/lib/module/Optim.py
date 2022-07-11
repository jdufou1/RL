"""
Optim : optimizer
DUFOURMANTELLE JEREMY
"""

import numpy as np

from lib.module.Linear import Linear

from tqdm import tqdm

class Optim :
    def __init__(self,net, loss, eps=1e-3):
        self.net=net
        self.loss=loss
        self.eps=eps

    def step(self,batch_x,batch_y):
        pass_forward=self.net.forward(batch_x)
        loss = self.loss.forward(batch_y,pass_forward).mean()
        backward_loss=self.loss.backward(batch_y,pass_forward)
        self.net.backward_delta(batch_x,backward_loss)
        self.net.backward_update_gradient(batch_x,backward_loss)
        self.net.update_parameters(self.eps)
        self.net.zero_grad()
        return loss

    def update(self):
        """abstract method"""
        pass

class SGD(Optim):
    
    def __init__(self, net, loss,datax,datay,batch_size=20,nbIter=100, eps=1e-3):
        self.net=net
        self.loss=loss
        self.eps=eps
        self.datax=datax
        self.datay=datay
        self.batch_size=batch_size
        self.nbIter=nbIter

    def creation_dataset_minibatch(self):
        size=len(self.datax)
        values = np.arange(size)
        np.random.shuffle(values)
        nb_batch = size // self.batch_size
        if (size % self.batch_size != 0):
            nb_batch += 1
        for i in range(nb_batch):
            index=values[i * self.batch_size:(i + 1) * self.batch_size]
            yield self.datax[index],self.datay[index]

    def update(self):
        list_loss = []
        for _ in tqdm(range(self.nbIter)):
            list_loss_batch = []
            for batch_x,batch_y in self.creation_dataset_minibatch():
                list_loss_batch.append( self.step(batch_x,batch_y) )
            list_loss.append(np.array(list_loss_batch).mean())
        return list_loss


class AdaGrad(Optim):

    def __init__(self, net, loss,datax,datay,batch_size=20,nbIter=100, eps=1e-3):
        self.net=net
        self.loss=loss
        self.datax=datax
        self.datay=datay
        self.batch_size=batch_size
        self.nbIter=nbIter
        
        self.eps = eps
        self.G = {}
        self.Biais = {}
        for module in self.net._modules.values():
            if module._parameters is not None :
                self.G[module] = np.zeros(module._parameters.shape)
            if isinstance(module,Linear) :
                self.Biais[module] = np.zeros(module._gradbiais.shape)

    def step(self,batch_x,batch_y):
        pass_forward=self.net.forward(batch_x)
        loss = self.loss.forward(batch_y,pass_forward).mean()
        backward_loss=self.loss.backward(batch_y,pass_forward)
        self.net.backward_delta(batch_x,backward_loss)
        self.net.backward_update_gradient(batch_x,backward_loss)
        
        # Update of the learning rate and the parameters

        for module in self.net._modules.values():
            # module.update_parameters(self.eps)
            if module._parameters is not None :
                self.G[module] += (module._gradient ** 2)
                lr_params = self.eps / (np.sqrt(self.G[module] + 1e-8)) 
                if isinstance(module,Linear) :
                    module._parameters -= lr_params * module._gradient
                    if(module._biais_active):
                        self.Biais[module] += (module._gradbiais ** 2)
                        lr_biais = self.eps / (np.sqrt(self.Biais[module] + 1e-8)) 
                        module._biais -= lr_biais*module._gradbiais
                else :
                    module.update_parameters(lr_params)
        
        # self.net.update_parameters(self.eps)
        self.net.zero_grad()
        return loss

    def creation_dataset_minibatch(self):
        size=len(self.datax)
        values = np.arange(size)
        np.random.shuffle(values)
        nb_batch = size // self.batch_size
        if (size % self.batch_size != 0):
            nb_batch += 1
        for i in range(nb_batch):
            index=values[i * self.batch_size:(i + 1) * self.batch_size]
            yield self.datax[index],self.datay[index]

    def update(self):
        list_loss = []
        for _ in tqdm(range(self.nbIter)):
            list_loss_batch = []
            for batch_x,batch_y in self.creation_dataset_minibatch():
                list_loss_batch.append( self.step(batch_x,batch_y) )
            list_loss.append(np.array(list_loss_batch).mean())
        return list_loss
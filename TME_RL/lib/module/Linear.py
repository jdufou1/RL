"""
Linear module
Jérémy DUFOURMANTELLE
"""

import numpy as np

from lib.module.Module import Module

class Linear(Module):

    def __init__(self, input_size, output_size, biais_active=True):
        self._input_size = input_size
        self._output_size = output_size
        self._parameters =  2 * (np.random.rand(input_size, output_size) - 0.5)
        self._gradient = np.zeros((input_size, output_size))
        self._biais_active = biais_active

        if (self._biais_active):
            self._biais = 2 * (np.random.randn(output_size) - 0.5)
            self._gradbiais = np.zeros(output_size)

    def zero_grad(self):
        self._gradient = np.zeros(self._gradient.shape)

        if (self._biais_active):
            self._gradbiais = np.zeros(self._gradbiais.shape)

    def forward(self, X):
        self._forward = np.dot(X,self._parameters)

        if(self._biais_active):
                self._forward = np.add(self._forward,self._biais)

        return self._forward

    def update_parameters(self, learning_rate=1e-3):
        
        self._parameters -= learning_rate*self._gradient

        if(self._biais_active):
            self._biais -= learning_rate*self._gradbiais

    def backward_update_gradient(self, input, delta):
        
        self._gradient = np.dot(input.T, delta)

        if (self._biais_active):
            self._gradbiais = np.sum(delta,axis=0)

    def backward_delta(self, input, delta):
        
        self._delta = np.dot(delta,self._parameters.T)
        
        return self._delta
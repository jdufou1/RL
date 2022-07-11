"""
Softmax : activation function
Jérémy DUFOURMANTELLE
"""

import numpy as np

from lib.module.Module import Module

class Softmax(Module):

    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, X):
        sum_exp = np.sum(np.exp(X), axis=1,keepdims=True)
        self._forward = np.exp(X)/sum_exp
        return self._forward

    def update_parameters(self, learning_rate=1e-3):
        pass

    def backward_delta(self, input, delta):
        sum_exp = np.sum(np.exp(input), axis=1,keepdims=True)
        z = np.exp(input)/sum_exp
        derive = z * (1 - z)
        self._delta = delta * derive
        return self._delta
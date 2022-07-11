"""
TanH : activation function
Jérémy DUFOURMANTELLE
"""

import numpy as np

from lib.module.Module import Module

class TanH(Module):

    def __init__(self):
        super(TanH, self).__init__()

    def forward(self, X):
        self._forward=np.tanh(X)
        return self._forward

    def update_parameters(self, learning_rate=1e-3):
        pass

    def backward_delta(self, input, delta):
        derivative = (1-(np.tanh(input)**2))
        self._delta = delta * derivative
        return self._delta
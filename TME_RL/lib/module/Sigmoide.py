"""
Sigmoide : activation function
Jérémy DUFOURMANTELLE
"""

import numpy as np

from lib.module.Module import Module

class Sigmoide(Module):

    def __init__(self):
        super(Sigmoide, self).__init__()

    def forward(self, X):
        self._forward = 1 / (1 + np.exp(-X))
        return self._forward

    def update_parameters(self, learning_rate=1e-3):
        pass

    def backward_delta(self, input, delta):
        derivative = 1 / (1 + np.exp(-input) )
        self._delta = delta * ( derivative * ( 1 - derivative) )
        return self._delta
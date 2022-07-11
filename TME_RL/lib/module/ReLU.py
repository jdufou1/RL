"""
Classe ABSTRAITE ReLU.
ABITBOL YOSSEF
DUFOURMANTELLE JEREMY
"""

from lib.module.Module import Module

import numpy as np

class ReLU(Module):

    def __init__(self):
        super(ReLU, self).__init__()


    def forward(self, X):
        """Calcule la passe forward"""
        self._forward =  np.where(X>self._threshold,X,0.)
        return self._forward

    def update_parameters(self, learning_rate=1e-3):
        pass

    def backward_delta(self, input, delta):
        """Calcul la derivee de l'erreur"""
        derive = (input > self._threshold).astype(float)
        self._delta= delta * derive
        return self._delta
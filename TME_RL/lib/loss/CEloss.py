"""
Cross Entropique : loss function
Jérémy DUFOURMANTELLE
"""

import numpy as np

from lib.loss.Loss import Loss

class CEloss(Loss):

    def forward(self, y, yhat):

        return 1 - np.sum(yhat * y, axis = 1)

    def backward(self, y, yhat):

        return yhat - y
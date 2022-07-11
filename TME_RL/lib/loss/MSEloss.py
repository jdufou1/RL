"""
Mean Squared Error : loss function
Jérémy DUFOURMANTELLE
"""

import numpy as np

from lib.loss.Loss import Loss

class MSEloss(Loss):

    def forward(self, y, yhat):

        return np.sum((y-yhat)**2,axis=1,keepdims=True)

    def backward(self, y, yhat):

        return 2*(yhat-y)
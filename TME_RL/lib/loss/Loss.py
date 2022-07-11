"""
Abstract class : Loss
Jérémy DUFOURMANTELLE
"""

class Loss(object):

    def forward(self, y, yhat):
        """Compute the cost with two inputs"""
        pass

    def backward(self, y, yhat):
        """Compute the cost gradient with prediction yhat"""
        pass
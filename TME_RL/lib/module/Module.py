"""
Abstract class : Module
Jérémy DUFOURMANTELLE
"""

class Module(object):
    def __init__(self):
        """Initialisation of the parameters"""
        self._parameters = None
        self._gradient = None
        self._forward = None
        self._delta = None

    def zero_grad(self):
        """gradient cancelling"""
        pass

    def forward(self, X):
        """compute forward"""
        pass
    
    def update_parameters(self, learning_rate=1e-3):
        """Compute the parameters update with gradient and learning rate"""
        self._parameters -= learning_rate*self._gradient

    def backward_update_gradient(self, input, delta):
        """gradient update"""
        pass

    def backward_delta(self, input, delta):
        """compute the derivative of the error"""
        pass
"""
Sequential module
DUFOURMANTELLE JEREMY
"""

from collections import OrderedDict

from lib.module.Module import Module

class Sequential(Module):

    def __init__(self, *args):
        super(Sequential, self).__init__()
        self._modules = OrderedDict()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for cle, module in args[0].items():
                 self._modules[cle] = module
        else:
            for index, module in enumerate(args):
                self._modules[str(index)] = module

    def zero_grad(self):
        for module in self._modules.values():
            module.zero_grad()

    def forward(self, X):
        input=X
        for module in self._modules.values():
            input = module.forward(input)
        self._forward=input
        return self._forward

    def update_parameters(self, learning_rate=1e-3):
        for module in self._modules.values():
            module.update_parameters(learning_rate)

    def backward_update_gradient(self, input, delta):
        modules=list(self._modules.values())[::-1]
        suiv = modules[0]
        for module in modules[1:]:
            prec=module
            suiv.backward_update_gradient(prec._forward,delta)
            delta=suiv._delta
            suiv=prec
        suiv.backward_update_gradient(input, delta)

    def backward_delta(self, input, delta):
        modules = list(self._modules.values())[::-1]
        suiv = modules[0]
        for module in modules[1:]:
            prec = module
            delta = suiv.backward_delta(prec._forward, delta)
            suiv = prec
        self._delta = suiv.backward_delta(input,delta)
        return self._delta
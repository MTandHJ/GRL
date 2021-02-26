


import torch
import eagerpy as ep
from foolbox.attacks import LinfProjectedGradientDescentAttack


def bce_loss(probs, targets):
    loss = targets * probs.log() + (1 - targets) * (1 - probs).log()
    return loss.mean()

class LinfPGDBCE(LinfProjectedGradientDescentAttack):

    def get_loss_fn(self, model, targets):
        def loss_fn(inputs):
            preds = model(inputs)
            return bce_loss(preds, targets)
        return loss_fn




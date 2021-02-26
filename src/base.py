


import torch
import torch.nn as nn
import math
import foolbox as fb
import eagerpy as ep
from .utils import AverageMeter, ProgressMeter
from .criteria import LogitsAllFalse
from .loss_zoo import cross_entropy, bce_loss




class Coach:

    def __init__(
        self, mc, md, *, 
        device, leverage,
        normalizer, optimizer, 
        learning_policy   
    ):
        self.mc = mc
        self.md = md
        self.device = device
        self.leverage = leverage
        self.normalizer = normalizer
        self.optimizer = optimizer
        self.learning_policy = learning_policy
        self.loss = AverageMeter("Loss")
        self.acc1 = AverageMeter("Acc_cls")
        self.acc2 = AverageMeter("Acc_dom")
        self.progress = ProgressMeter(self.loss, self.acc1, self.acc2)
    
    def adjust_weight(self, cur_epoch, gamma=10.):
        return (2 / (1 + math.exp(-gamma * cur_epoch)) - 1) * self.leverage

    def save(self, path):
        torch.save(self.mc.state_dict(), path + "/paras.pt")

    def adv_train(self, trainlaoder, attacker, *, epsilon=None, epoch=8888):
        self.progress.step()
        for inputs_clean, labels in trainlaoder:
            inputs_clean = inputs_clean.to(self.device)
            labels_cls = labels.to(self.device)
            labels_dom = torch.zeros(labels.size(0), device=self.device)

            _, inputs_adv, _ = attacker(inputs_clean, LogitsAllFalse(labels_dom), epsilon)
            inputs = torch.cat((inputs_clean, inputs_adv), dim=0)
            labels = torch.cat((labels_dom, 1 - labels_dom), dim=0)

            self.mc.train()
            self.md.train()
            outs_c = self.mc(self.normalizer(inputs_clean))
            outs_d = self.md(self.normalizer(inputs))

            leverage = self.adjust_weight(cur_epoch=epoch)
            loss1 = cross_entropy(outs_c, labels_cls, reduction="mean")
            loss2 = bce_loss(outs_d, labels, reduction="mean")
            loss = loss1 + loss2 * leverage

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.loss.update(loss.item(), n=inputs_clean.size(0), mode="mean")
            self.acc1.update(outs_c.argmax(-1).eq(labels_cls).sum(), n=inputs_clean.size(0), mode="sum")
            self.acc2.update(outs_d.floor().eq(labels).sum(), n=inputs.size(0), mode="sum")

        self.progress.display(epoch=epoch)
        self.learning_policy.step()
        return self.loss.avg





class Adversary:
    """
    Adversary is mainly based on foolbox, especially pytorchmodel.
    model: Make sure that the model's output is the logits or the attack is adapted.
    attacker: the attack implemented by foolbox or a similar one
    device: ...
    bounds: typically [0, 1]
    preprocessing: including mean, std, which is similar to normalizer
    criterion: typically given the labels and consequently it is Misclassification, 
            other critera could be given to carry target attack or black attack.
    """
    def __init__(
        self, model, attacker, device,
        bounds, preprocessing, epsilon
    ):
        model.eval()
        self.fmodel = fb.PyTorchModel(
            model,
            bounds=bounds,
            preprocessing=preprocessing,
            device=device
        )
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.attacker = attacker 

    def attack(self, inputs, criterion, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        self.model.eval() # make sure in evaluating mode ...
        return self.attacker(self.fmodel, inputs, criterion, epsilons=epsilon)

    @torch.no_grad()
    def accuracy(self, inputs, labels):
        inputs_, labels_ = ep.astensors(inputs, labels)
        del inputs, labels

        self.model.eval() # make sure in evaluating mode ...
        predictions = self.fmodel(inputs_).argmax(axis=-1)
        accuracy = (predictions == labels_)
        return accuracy.sum().item()

    def success(self, inputs, criterion, epsilon=None):
        _, _, is_adv = self.attack(inputs, criterion, epsilon)
        return is_adv.sum().item()

    def evaluate(self, dataloader, epsilon=None):
        datasize = len(dataloader.dataset)
        running_accuracy = 0
        running_success = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            running_accuracy += self.accuracy(inputs, labels)
            running_success += self.success(inputs, labels, epsilon)
        return running_accuracy / datasize, running_success / datasize

    def __call__(self, inputs, criterion, epsilon=None):
        return self.attack(inputs, criterion, epsilon)
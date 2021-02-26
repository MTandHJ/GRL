



import torch.nn as nn




class MC(nn.Module):

    def __init__(self, gf, gc):
        super(MC, self).__init__()

        self.gf = gf
        self.gc = gc


    def forward(self, inputs):
        features = self.gf(inputs)
        outs = self.gc(features)
        return outs


class MD(nn.Module):

    def __init__(self, gf, gd):
        super(MD, self).__init__()

        self.gf = gf
        self.gd = gd

    def forward(self, inputs):
        features = self.gf(inputs)
        outs = self.gd(features)
        return outs



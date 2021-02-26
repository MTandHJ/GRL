
import torch
from torch.autograd import Function



class RevGrad(Function):

    @staticmethod
    def forward(ctx, inputs):
        return inputs

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs.neg()



if __name__ == "__main__":

    test = RevGrad.apply
    x = torch.tensor([1., 2.], requires_grad=True)
    z = test(x)
    y = z.sum()
    y.backward()
    print(x.grad)

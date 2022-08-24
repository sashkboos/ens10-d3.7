# Implementation of the LeNet-Style baseline (see https://www.sciencedirect.com/science/article/pii/S0022169421013512)
# The input is a grid with 2*n channels where n is the number of variables in the pressure levels
# The output of the network are two grids

from torch import nn as nn
import torch
import torch.nn.functional as F

BASE_FILTER = 16  # has to be divideable by 4, originally 16

def batch_norm(
    out_channels, dim=2
):  # Does not have the same parameters as the original batch normalization used in the tensorflow 1.14 version of this project
    if dim == 2:
        bnr = torch.nn.BatchNorm2d(out_channels)
    else:
        bnr = torch.nn.BatchNorm3d(out_channels)
    return bnr

# 3x3x3 convolution module
def Conv(in_channels, out_channels, filter_sizes=3, dim=2):
    if dim == 2:
        return nn.Conv2d(
            in_channels, out_channels, filter_sizes, padding=0
        )
    else:
        return nn.Conv3d(
            in_channels, out_channels, filter_sizes, padding=(filter_sizes - 1) // 2
        )

# Activation function
def activation(x):
    # Simple relu
    return F.elu(x, inplace=True)

# Conv Batch Relu module
class ConvBatchElu(nn.Module):
    def __init__(self, in_channels, out_channels, filter_sizes=3):
        super(ConvBatchElu, self).__init__()
        self.conv = Conv(in_channels, out_channels, filter_sizes)
        self.bnr = batch_norm(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bnr(activation(x))
        return x


class LeNet(nn.Module):
    def __init__(
            self, in_channels=14, args=None
    ):
        super(LeNet, self).__init__()

        self.dim = 2

        self.ic = in_channels
        oc = 2
        self.conv1 = ConvBatchElu(self.ic, 64)
        self.conv2 = ConvBatchElu(64, 32)
        self.conv3 = ConvBatchElu(32, 16)

        self.dense1 = nn.Linear(19, 6)
        self.dense2 = nn.Linear(6,2)

        self.pad = nn.ReplicationPad2d(3)
        self.unfold = nn.Unfold((7,7))

    def forward(self, inp):

        x = inp[:,:self.ic,:,:]
        emb = inp[:,self.ic:,:,:]

        x = self.pad(x)
        x = self.unfold(x)

        x = torch.movedim(x, -2, -1)
        x = torch.reshape(x, (x.shape[1], self.ic, 7, 7))

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x[:,:,0,0]

        emb = torch.movedim(emb, 1, -1)
        emb = emb.reshape((emb.shape[-3] * emb.shape[-2]),3)
        x = torch.concat((emb, x), dim=1)
        

        x = self.dense1(x)
        output = self.dense2(x)


        output = torch.reshape(output, (inp.shape[0],inp.shape[-2]*inp.shape[-1],2))
        return output

def LeNet_prepare(args):

    if args.target_var in ['t2m']:
        return LeNet(22, args)
    return LeNet(14, args)
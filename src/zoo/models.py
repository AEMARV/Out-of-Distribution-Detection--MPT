import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models.resnet as ResNet
import torch.nn.functional as F
from src.layers import Fold
from src.layers import Sampler
from src.layers import *

class SimpleCIFAR10(nn.Module):

    def __init__(self,layers=12,classnum=-1,num_filter=64,filter_scale=1):
        super().__init__()
        import math
        num_filter = math.ceil(num_filter * filter_scale)
        layer = torch.nn.Conv2d(3, num_filter, (3, 3), stride=(1, 1), padding=0)
        self.convs = [layer]
        self.add_module('conv_'+'1',layer)
        for i in range(layers-2):
            layer = torch.nn.Conv2d(num_filter*2, num_filter, (3,3), stride=(1,1), padding=0)
            self.add_module('conv_'+str(i+2),layer)
            self.convs = self.convs +[layer]

        layer = torch.nn.Conv2d(num_filter*2, classnum, (3,3), stride=(1,1), padding=0)
        self.add_module('conv_' + str(layers), layer)
        self.convs = self.convs +[layer]


        # self.add_module('conv1',self.conv1)
        # self.add_module('conv2', self.conv2)
        # self.add_module('conv3', self.conv3)
        # self.add_module('conv4', self.conv4)

    def forward(self,x):
        nl = Fold()
        for i in range(self.convs.__len__()-1):
            x = self.convs[i](x)
            x = nl(x)
        x= self.convs[-1](x)
        x = F.max_pool2d(x,x.shape[2],stride=1)
        return x

class FCSimpleCIFAR10(nn.Module):

    def __init__(self,layers=12,classnum=-1,num_filter=64,filter_scale=1):
        super().__init__()
        num_filter = num_filter * filter_scale
        layer = torch.nn.Conv2d(3, num_filter, (3, 3), stride=(1, 1), padding=0)
        self.convs = [layer]
        self.fcs= []
        self.add_module('conv_'+'1',layer)
        for i in range(layers-1):
            layer = torch.nn.Conv2d(num_filter*2, num_filter, (3,3), stride=(1,1), padding=0)
            self.add_module('conv_'+str(i+2),layer)
            self.convs = self.convs +[layer]


        layer = torch.nn.Conv2d(num_filter*2, num_filter, (1, 1), stride=(1, 1), padding=0)
        self.add_module('fc_' + str(1), layer)
        self.fcs = self.fcs + [layer]

        layer = torch.nn.Conv2d(num_filter*2, classnum, (1, 1), stride=(1, 1), padding=0)
        self.add_module('fc_' + str(2), layer)
        self.fcs = self.fcs + [layer]


    def forward(self,x):

        nl = Fold()
        for i in range(self.convs.__len__()-1):
            x = self.convs[i](x)
            x = nl(x)
        x= self.convs[-1](x)
        x = F.max_pool2d(x,x.shape[2],stride=1)
        for i in range(self.fcs.__len__()-1):
            x= self.fcs[i](x)
            x= nl(x)
        x= self.fcs[-1](x)

        return x

class ProbabilisticCNN(nn.Module):

    def __init__(self,layers=12,classnum=-1,num_filter=64,filter_scale=1):
        super().__init__()
        num_filter = num_filter * filter_scale
        layer = torch.nn.Conv2d(3, num_filter, (3, 3), stride=(1, 1), padding=0)
        self.convs = [layer]
        self.fcs= []
        self.add_module('conv_'+'1',layer)
        self.input_noise_layer= UniformNoise((1,3,1,1))
        self.add_module('in_noise',self.input_noise_layer)
        for i in range(layers-1):
            layer = torch.nn.Conv2d(num_filter*2, num_filter, (3,3), stride=(1,1), padding=0)
            layer.weight.data = layer.weight.data *2
            self.add_module('conv_'+str(i+2),layer)
            self.convs = self.convs +[layer]


        layer = torch.nn.Conv2d(2*num_filter, classnum, (3, 3), stride=(1, 1), padding=0)
        # self.add_module('fc_' + str(1), layer)
        self.add_module('conv_final',layer)
        self.convs = self.convs + [layer]

        # layer = torch.nn.Conv2d(num_filter*2, classnum, (1, 1), stride=(1, 1), padding=0)
        # self.add_module('fc_' + str(2), layer)
        # self.fcs = self.fcs + [layer]


    def forward(self,x):

        nl = Fold()
        logprob= 0
        x,logprob = self.input_noise_layer(x)
        for i in range(self.convs.__len__()-1):
            x = self.convs[i](x)
            x = nl(x)
        x= self.convs[-1](x)
        x = F.max_pool2d(x,x.shape[2],stride=1)
        # x = x.log_softmax(dim=1)
        return x,logprob

class DPN_CIFAR10FC(nn.Module):

    def __init__(self,layers=12,classnum=-1,num_filter=64,filter_scale=1):
        super().__init__()
        num_filter = num_filter * filter_scale
        layer = torch.nn.Conv2d(3, num_filter, (3, 3), stride=(1, 1), padding=0)
        self.convs = [layer]

        self.fcs= []
        self.add_module('conv_'+'1',layer)
        self.nl = Sampler()
        self.add_module('nl',self.nl)
        for i in range(layers-1):
            layer = torch.nn.Conv2d(num_filter, num_filter, (3,3), stride=(1,1), padding=0)
            self.add_module('conv_'+str(i+2),layer)
            self.convs = self.convs +[layer]


        layer = torch.nn.Conv2d(num_filter, num_filter, (1, 1), stride=(1, 1), padding=0)
        self.add_module('fc_' + str(1), layer)
        self.fcs = self.fcs + [layer]

        layer = torch.nn.Conv2d(num_filter, classnum, (1, 1), stride=(1, 1), padding=0)
        self.add_module('fc_' + str(2), layer)
        self.fcs = self.fcs + [layer]


    def forward(self,x):

        energy = 0
        for i in range(self.convs.__len__()-1):
            x = self.convs[i](x)
            x,energy_temp = self.nl(x)
            energy = energy+energy_temp
        x= self.convs[-1](x)
        x = F.avg_pool2d(x,x.shape[2],stride=1)
        for i in range(self.fcs.__len__()-1):
            x= self.fcs[i](x)
            x, energy_temp = self.nl(x)
            energy = energy + energy_temp
        x= self.fcs[-1](x)

        return x,energy


class DPN_SimpleCIFAR10(nn.Module):

    def __init__(self,layers=12,classnum=-1,num_filter=64,filter_scale=1):
        super().__init__()
        import math
        num_filter = math.ceil(num_filter * filter_scale)
        layer = torch.nn.Conv2d(3, num_filter, (3, 3), stride=(1, 1), padding=0)
        self.convs = [layer]
        self.add_module('conv_'+'1',layer)
        self.nl = Sampler()
        self.add_module('nl', self.nl)

        for i in range(layers-2):
            layer = torch.nn.Conv2d(num_filter, num_filter, (3,3), stride=(1,1), padding=0)
            self.add_module('conv_'+str(i+2),layer)
            self.convs = self.convs +[layer]

        layer = torch.nn.Conv2d(num_filter, classnum, (3,3), stride=(1,1), padding=0)
        self.add_module('conv_' + str(layers), layer)
        self.convs = self.convs +[layer]

    def forward(self,x):
        energy=0
        for i in range(self.convs.__len__()-1):
            x = self.convs[i](x)
            x, energy_temp = self.nl(x)
            energy = energy + energy_temp
        x = self.convs[-1](x)
        x = F.max_pool2d(x,x.shape[2],stride=1)
        x = x.log_softmax(dim=1)
        # x, energy_temp = self.nl(x,numstates=x.shape[1])
        # energy = energy + energy_temp
        return x , energy
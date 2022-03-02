import torch
from torch.nn import Module
from src.utils import softmax
class Fold(Module):
    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)

    def forward(self,x):
        xp = x.relu()
        xn = (-x).relu()
        x= torch.cat([xp,xn],dim=1)
        return x

class GrayToRGB(Module):
    ''' Dataloader module'''
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        if x.shape[0]==1:
            x= torch.cat([x,x,x],dim=0)
        return x

class Sampler(Module):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def sample(self,energy:torch.Tensor ,dim,beta=1):
        energy = energy.transpose(0,dim)
        lprob = (energy*beta).log_softmax(dim=0)
        energy = lprob
        prob = lprob.exp()
        cumprob = prob.cumsum(dim=0)
        randoms = torch.rand_like(prob[0:1,0:])
        sample = (cumprob > randoms).float()
        sample = (sample.cumsum(dim=0)==1).float().detach()

        energy = sample * lprob
        lprob = sample* lprob
        lprob[lprob!= lprob]=0
        energy[energy != energy] = 0
        energy = lprob
        energy = energy.sum(dim=0,keepdim=True)
        energy = energy.transpose(0,dim)
        sample = sample.transpose(0,dim)
        return sample, energy
    def sigmoid_sample(self,energy):
        prob = energy.sigmoid().detach()
        samples = prob> torch.rand_like(prob)
        energy = energy * (samples-0.5)
        samples = torch.cat([samples])
    def forward(self,x,numstates=2):
        init_shape = x.shape
        x = x.reshape([x.shape[0],numstates,-1,x.shape[2],x.shape[3]])
        if self.training:
            sample, energy = self.sample(x,1,beta=1)
        else:
            sample, energy = self.sample(x, 1, beta=1000)
        sample = sample.reshape(init_shape)
        energy = energy.squeeze(1)
        energy = energy.sum(dim=[2,3],keepdim=True).sum(dim=1,keepdim=True)
        return sample, energy

class UniformNoise(Module):
    def __init__(self,input_shape,prior_range=(-1,1),numbits=8):
        super().__init__()
        self.lprob =torch.nn.Parameter( torch.ones(input_shape,requires_grad=True))
        self.register_parameter('lprob',self.lprob)
        self.prior_range = prior_range
        self.numbits= numbits
    def forward(self,x):
        if not self.training:
            # print('working')
            return x,0
        select_prob = self.lprob.sigmoid()
        select = (torch.rand_like(x)<=select_prob)
        noise = torch.rand_like(x)*(self.prior_range[1] - self.prior_range[0])
        noise = noise + self.prior_range[0]
        x = select * x + (1-select.float()) * noise
        select_lprob = softmax(self.lprob,torch.zeros_like(self.lprob))
        deselect_lprob = softmax(-self.lprob,torch.zeros_like(self.lprob))# - self.numbits
        lprob= select_lprob *select.float() + deselect_lprob *(1-select.float())
        return x, lprob.sum(dim=(1,2,3),keepdim=True)



class NormalNoise(Module):
    def __init__(self,input_shape,prior_range=(-1,1),numbits=8):
        super().__init__()
        self.weight =torch.nn.Parameter(1* torch.ones(input_shape,requires_grad=True))
        self.register_parameter('weight',self.weight)
        self.prior_range = prior_range
        self.numbits= numbits
    def forward(self,x):
        if not self.training:
            # print('working')
            return x,0
        weight = self.weight/1
        print(weight)
        noise = (torch.randn_like(x)/weight).detach()
        lprob = -((noise*weight)**2)/2# - self.weight.abs().log()/2
        x = x + noise
        return x, lprob.sum(dim=(1,2,3),keepdim=True)






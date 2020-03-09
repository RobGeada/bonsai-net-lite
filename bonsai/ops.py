import torch.nn as nn
import torch
import math
import numpy as np
from torch.autograd import Variable


# === OPERATION HELPERS ================================================================================================
def bracket(ops, ops2=None):
    out = ops + [nn.BatchNorm2d(ops[-1].out_channels, affine=True)]
    if ops2:
        out += [nn.ReLU(inplace=False)] + ops2 + [nn.BatchNorm2d(ops[-1].out_channels, affine=True)]
    return nn.Sequential(*out)


# from https://github.com/quark0/darts/blob/master/cnn/utils.py
def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


# === INDIVIDUAL OPERATIONS ============================================================================================
class MaxPool(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.op = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
    
    def forward(self,x):
        return self.op(x)


class SingleConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding):
        super().__init__()
        if c_in == c_out:
            self.op = bracket([
                nn.Conv2d(c_in, c_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=c_in, bias=False),
            ])
        else:
            self.op = bracket([
                nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
            ])

    def forward(self, x):
        return self.op(x)


class DilatedConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding):
        super().__init__()
        self.op = bracket([
            nn.Conv2d(c_in, c_in, kernel_size, stride=stride, dilation=2, padding=padding, groups=c_in, bias=False),
            nn.Conv2d(c_in, c_in, 1, padding=0, bias=False)
        ])

    def forward(self, x):
        return self.op(x)


class SeparableConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding):
        super().__init__()

        self.op = bracket([
            nn.Conv2d(c_in, c_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=c_in, bias=False),
            nn.Conv2d(c_in, c_in, kernel_size=1, padding=0, bias=False)
        ], [
            nn.Conv2d(c_in, c_in, kernel_size=kernel_size, stride=1, padding=padding, groups=c_in, bias=False),
            nn.Conv2d(c_in, c_in, kernel_size=1, padding=0, bias=False)
        ] )

    def forward(self, x):
        return self.op(x)


class Scaler(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.op = nn.Conv2d(self.c_in, self.c_out, kernel_size=1, stride=1)

    def forward(self, x):
        return self.op(x)
        #return torch.cat([x]*self.c_out/self.c_in,dim=1)


class MinimumIdentity(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.out_channels = c_out

        if stride == 1:
            self.strider = nn.Sequential()
        else:
            self.strider = MaxPool(kernel_size=3, stride=stride, padding=padsize(k=3, s=stride))

        if c_in == c_out:
            self.scaler = nn.Sequential()
        else:
            self.scaler = Scaler(c_in, c_out)

    def forward(self, x):
        return self.strider(self.scaler(x))


def dim_mod(dim, by_c, by_s):
    return dim[0], int(dim[1]*by_c), dim[2]//by_s, dim[3]//by_s


class Zero(nn.Module):
    def __init__(self, stride, upscale=1):
        super().__init__()
        self.stride = stride
        self.upscale = upscale
        if stride == 1 and upscale == 1:
            self.op = lambda x: torch.zeros_like(x)
        else:
            self.op = lambda x: torch.zeros(dim_mod(x.shape, upscale, stride),
                                            device=torch.device('cuda'),
                                            dtype=x.dtype)

    def forward(self, x):
        return self.op(x)


class NNView(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)


# === PRUNERS ==========================================================================================================
class Pruner(nn.Module):
    def __init__(self, m=1e5, mem_size=0, init=None):
        super().__init__()
        if init is None:
            init = .1
        self.mem_size = mem_size
        self.weight = nn.Parameter(torch.tensor([init]))
        self.m = m

        #self.gate = lambda w: torch.sigmoid(self.m*w)
        self.gate = lambda w: (.5 * w / torch.abs(w)) + .5
        self.saw = lambda w: (self.m * w - torch.floor(self.m * w)) / self.m
        self.weight_history = [1]

    def __str__(self):
        return 'Pruner: M={},N={}'.format(self.M, self.channels)

    def num_params(self):
        # return number of differential parameters of input model
        return sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, self.parameters())])

    def track_gates(self):
        self.weight_history.append(self.gate(self.weight).item())

    def get_deadhead(self, verbose=False):
        deadhead = not any(self.weight_history)
        if verbose:
            print(self.weight_history, deadhead)
        self.weight_history = []
        return deadhead

    def deadhead(self):
        for param in self.parameters():
            param.requires_grad = False

    def sg(self):
        return self.saw(self.weight) + self.gate(self.weight)

    def forward(self, x):
        return self.sg() * x


class PrunableOperation(nn.Module):
    def __init__(self, op_function, name, mem_size, c_in, stride, pruner_init=None, prune=True):
        super().__init__()
        self.op_function = op_function
        self.stride = stride
        self.name = name
        self.op = self.op_function(c_in, stride)
        self.zero = name == 'Zero'
        self.prune = prune
        if self.prune:
            self.pruner = Pruner(mem_size=mem_size, init=pruner_init)

    def track_gates(self):
        self.pruner.track_gates()

    def deadhead(self):
        if self.zero or not self.pruner.get_deadhead():
            return 0
        else:
            self.op = Zero(self.stride)
            self.zero = True
            self.pruner.deadhead()
            return 1

    def __str__(self):
        return self.name

    def forward(self, x):
        if self.prune:
            out = self.op(x) if self.zero else self.pruner(self.op(x))
        else:
            out = self.op(x)
        return out


class PrunableInputs(nn.Module):
    def __init__(self, dims, scale_mod, genotype, random_ops, prune=True):
        super().__init__()
        # weight inits
        if genotype.get('weights') is None:
            pruner_inits = [None]*len(dims)
        else:
            pruner_inits = genotype.get('weights')

        # zero inits
        if genotype.get('zeros') is None:
            self.zeros = [False] * len(dims)
        else:
            self.zeros = genotype.get('zeros')

        if random_ops is not None:
            self.zeros = [False if np.random.rand()<(random_ops['i_c']/len(self.zeros)) else True for i in self.zeros]
            if all(self.zeros):
                self.zeros[np.random.choice(range(len(self.zeros)))]=False
        self.unified_dim = dims[-1]
        self.prune = prune
        ops, strides, upscales, pruners = [], [], [], []

        for i, dim in enumerate(dims):
            stride = self.unified_dim[1]//dim[1] if dim[3] != self.unified_dim[3] else 1
            strides.append(stride)
            c_in, c_out = dim[1], self.unified_dim[1]
            upscales.append(c_out/c_in)
            if self.zeros[i]:
                ops.append(Zero(stride, c_out/c_in))
            else:
                ops.append(MinimumIdentity(c_in, c_out, stride))
            if self.prune:
                pruners.append(Pruner(init=pruner_inits[i]))

        self.ops = nn.ModuleList(ops)
        self.strides = strides
        self.upscales = upscales
        if self.prune:
            self.pruners = nn.ModuleList(pruners)
        self.scaler = MinimumIdentity(self.unified_dim[1], self.unified_dim[1]*scale_mod, stride=1)

    def track_gates(self):
        [pruner.track_gates() for pruner in self.pruners]

    def deadhead(self):
        out = 0
        for i, pruner in enumerate(self.pruners):
            if self.zeros[i] or not pruner.get_deadhead():
                out += 0
            else:
                self.ops[i] = Zero(self.strides[i], self.upscales[i])
                self.zeros[i] = True
                pruner.deadhead()
                #self.pruners[i] = None
                out += 1
        return out

    def get_ins(self):
        return [i - 1 if i else 'In' for i, zero in enumerate(self.zeros) if not zero]

    def __str__(self):
        return str(self.get_ins())

    def forward(self, xs):
        if self.prune:
            out = sum([op(xs[i]) if self.zeros[i] else self.pruners[i](op(xs[i])) for i,op in enumerate(self.ops)])
        else:
            out = sum([op(xs[i]) for i, op in enumerate(self.ops)])
        return self.scaler(out)


# === NETWORK UTILITIES ===============================================================================================
class Classifier(nn.Module):
    def __init__(self, position, preserve_aux, in_size, out_size, scale=False):
        super().__init__()
        self.position = position
        self.preserve_aux = preserve_aux
        if 0:#scale:
            self.scaler = nn.MaxPool2d(3, stride=2, padding=padsize(s=2))
            in_size/=np.array([1,1,2,2])
        else:
            self.scaler = nn.Sequential()
        self.op = nn.Sequential(
            NNView(),
            nn.Linear(int(np.prod(in_size[1:])), out_size)
        )

    def forward(self, x):
        return self.op(self.scaler(x))


def padder(c_in, c_out, stride=1):
    return MinimumIdentity(c_in, c_out, stride=stride)


def initializer(c_in, c_out):
    return SingleConv(c_in, c_out, kernel_size=1, stride=1, padding=padsize(k=1, s=1))


def normalizer(c_in):
    return nn.BatchNorm2d(c_in, affine=True)


def padsize(s=1, k=3, d=1):
    pad = math.ceil((k * d - d + 1 - s) / 2)
    return pad


# === SEARCH SPACE =====================================================================================================
commons = {
    'Identity':     lambda c_in, s: MinimumIdentity(c_in, c_in, stride=s),
    'Avg_Pool_3x3': lambda c_in, s: nn.AvgPool2d(3, stride=s, padding=padsize(s=s)),
    'Max_Pool_3x3': lambda c_in, s: nn.MaxPool2d(3, stride=s, padding=padsize(s=s)),
    'Sep_Conv_3x3': lambda c_in, s: SeparableConv(c_in, c_in, 3, stride=s, padding=padsize(s=s)),
    'Sep_Conv_5x5': lambda c_in, s: SeparableConv(c_in, c_in, 5, stride=s, padding=padsize(k=5, s=s)),
    'Dil_Conv_3x3': lambda c_in, s: DilatedConv(c_in, c_in, 3, stride=s, padding=padsize(d=2, s=s)),
    'Dil_Conv_5x5': lambda c_in, s: DilatedConv(c_in, c_in, 5, stride=s, padding=padsize(d=2, k=5, s=s)),
}


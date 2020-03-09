from bonsai.ops import *
from bonsai.helpers import *


class Edge(nn.Module):
    def __init__(self, dim, origin, target, op_sizes, stride=1, genotype=None, allocation=None, prune=True):
        super().__init__()
        self.stride = stride
        if genotype is not None:
            available_ops = {k: commons[k] for (k, weight) in genotype}
            op_weights = [weight for (k, weight) in genotype]
        elif allocation is not None:
            available_ops = {k: commons[k] for k in allocation}
            self.used = sum([op_sizes[k] for k in allocation])
            self.possible = sum(op_sizes.values())
            op_weights = [1.]*len(available_ops)
        else:
            available_ops = commons
            self.used = sum([v for k, v in op_sizes.items()])
            self.possible = sum(op_sizes.values())
            op_weights = [None]*len(commons)

        self.ops = []
        for i, (key, op) in enumerate(available_ops.items()):
            prune_op = PrunableOperation(op_function=op,
                                         name=key,
                                         c_in=dim[1],
                                         mem_size=op_sizes[key],
                                         stride=stride,
                                         pruner_init=op_weights[i],
                                         prune=prune)
            self.ops.append(prune_op)
        self.ops = nn.ModuleList(self.ops)
        self.num_ops = len([op for op in self.ops if not op.zero])
        self.dim = dim
        self.out_dim = dim if self.stride == 1 else width_mod(dim, 2)
        self.origin = origin
        self.target = target
        
        if self.num_ops:
            self.normalizer = nn.BatchNorm2d(dim[1])
        self.zero = Zero(stride=self.stride)

    def deadhead(self):
        dhs = sum([op.deadhead() for i, op in enumerate(self.ops)])
        self.used = sum([op.pruner.mem_size for op in self.ops if not op.zero])
        self.num_ops -= dhs
        if self.num_ops == 0:
            self.zero = Zero(stride=self.stride)
        return dhs

    def genotype_compression(self, soft_ops=0, hard_ops=0):
        for op in self.ops:
            if not op.zero:
                hard_ops += op.pruner.mem_size
            soft_ops += op.pruner.sg() * op.pruner.mem_size if op.pruner else 0
        return soft_ops, hard_ops

    def __str__(self):
        return "{}->{}: {}, \t({:,} params)".format(
            self.origin,
            self.target,
            sorted([str(op) for op in self.ops if (not op.zero)]),
            general_num_params(self))

    def __repr(self):
        return str(self)


    def forward(self, x, drop_prob):
        if self.num_ops:
            outs = [op(x) if op.name in ['Identity','Zero'] else drop_path(op(x), drop_prob) for op in self.ops]
            return self.normalizer(sum(outs))
        else:
            return self.zero(x)
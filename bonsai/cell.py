from bonsai.ops import *
from bonsai.helpers import *
from bonsai.edge import Edge

import time
import numpy as np


def operation_allocate(aim_size, num, op_sizes):
    if num:
        alloc = [[(k, "{}_{}".format(k,i ), i, v) for k, v in op_sizes.items()] for i in range(num)]
        flat_alloc = [op for edge in alloc for op in edge]
        out = []
        while sum([x[-1] for x in out]) < aim_size:
            if not len(flat_alloc):
                break
            out_sum = sum([x[-1] for x in out])
            idx_size = np.inf
            min_size = min([x[-1] for x in flat_alloc])
            if aim_size < out_sum + min_size:
                break
            while aim_size < out_sum + idx_size:
                idx = np.random.choice([x[1] for x in flat_alloc], 1)
                idx_size = [x[-1] for x in flat_alloc if x[1] == idx][0]
            out += [x for x in flat_alloc if x[1] == idx]
            flat_alloc = [x for x in flat_alloc if x[1] != idx]
        output = [[x[0] for x in out if x[2] == i] for i in range(num)]
        return output
    else:
        return []



class Cell(nn.Module):
    def __init__(self, name, cell_type, dims, nodes, op_sizes, genotype=None, random_ops=False, prune=True):
        super().__init__()
        self.name = name
        self.cell_type = cell_type
        self.nodes = nodes
        self.dims = dims
        self.input_handler = PrunableInputs(dims,
                                            scale_mod=1 if self.cell_type is 'Normal' else 2,
                                            genotype={} if genotype is None else genotype['Y'],
                                            random_ops=random_ops,
                                            prune=prune)
        self.in_dim = channel_mod(dims[-1], dims[-1][1] if cell_type == 'Normal' else dims[-1][1]*2)
        self.scaler = MinimumIdentity(dims[-1][1],
                                      dims[-1][1] * (2 if self.cell_type is 'Reduction' else 1),
                                      stride=1)

        edges = []
        keys = {}

        # link input data to each antecedent node
        self.cell_size_est = 0
        stride=1 if cell_type == 'Normal' else 2
        if random_ops is not None:
            cell_cnx = sum([1 for origin in ['x', 'y'] for target in range(nodes)])
            cell_cnx += sum([1 for origin in range(nodes) for target in range(origin + 1, nodes)])

            strided_op_sizes = op_sizes[self.in_dim + (stride,)]
            strided_size = sum(strided_op_sizes.values())

            if stride==1:
                num_strided = cell_cnx
                num_unstrided = 0
                unstrided_size = 0
                unstrided_op_sizes = {}
            else:
                num_strided = 2*nodes
                unstrided_op_sizes = op_sizes[width_mod(self.in_dim,2)+(1,)]
                unstrided_size = sum(unstrided_op_sizes.values())
                num_unstrided = cell_cnx-num_strided

            strided_aim_size = num_strided*random_ops['e_c']*strided_size
            unstrided_aim_size = num_unstrided*random_ops['e_c']*unstrided_size
            strided_ops = operation_allocate(strided_aim_size, num_strided, strided_op_sizes)
            unstrided_ops = operation_allocate(unstrided_aim_size, num_unstrided, unstrided_op_sizes)
            allocation = (x for x in strided_ops+unstrided_ops)
        else:
            allocation = looping_generator([None])

        for origin in ['x', 'y']:
            for target in range(nodes):
                key = "{}->{}".format(origin, target)
                edges.append([key, Edge(self.in_dim,
                                        origin,
                                        target,
                                        stride=stride,
                                        allocation=next(allocation),
                                        op_sizes=op_sizes[self.in_dim+(stride,)],
                                        genotype=None if genotype is None else genotype.get(key),
                                        prune=prune)])
                keys[key] = {'origin': origin, 'target': target}
        self.in_dim = self.in_dim if cell_type == 'Normal' else width_mod(self.in_dim,2)

        # connect data nodes
        for origin in range(nodes):
            for target in range(origin+1, nodes):
                key = "{}->{}".format(origin, target)
                edges.append([key, Edge(self.in_dim,
                                        origin,
                                        target,
                                        allocation=next(allocation),
                                        op_sizes=op_sizes[self.in_dim+(1,)],
                                        genotype=None if genotype is None else genotype.get(key),
                                        prune=prune)])
                keys[key] = {'origin': origin, 'target': target}
        self.size_est = sum([edge[1].used for edge in edges])
        self.node_names = ['x', 'y']+list(range(self.nodes))
        self.normalizers = nn.ModuleDict({str(k): normalizer(self.in_dim[1]) for k in self.node_names})

        self.prune = prune
        self.edges = nn.ModuleDict(edges)
        self.key_ots = {k: (v['origin'], v['target']) for (k, v) in keys.items()}
        self.keys_by_origin = {i: [k for (k, v) in keys.items() if i == v['origin']] for i in self.node_names}
        self.keys_by_target = {i: [k for (k, v) in keys.items() if i == v['target']] for i in self.node_names}
        self.genotype_width = len(commons.items())

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, xs, drop_prob):
        start = time.time()
        raw_node_ins = {}
        start_mem = mem_stats(False)
        for node in self.node_names:
            if node=='x':
                raw_node_ins[node] = [self.scaler(xs[-1])]
            elif node=='y':
                raw_node_ins[node] = [self.input_handler(xs)]
            else:
                raw_node_ins[node] = []
    
        for node in self.node_names:   
            node_in = self.normalizers[str(node)](sum(raw_node_ins[node]))
            if node == self.nodes-1:
                return node_in
            for key in self.keys_by_origin[node]:
                raw_node_ins[self.key_ots[key][1]].append(self.edges[key](node_in, drop_prob))

    def genotype_compression(self, soft_ops=0, hard_ops=0, used_ratio=False):
        if not used_ratio:
            for key, edge in self.edges.items():
                soft_ops, hard_ops = edge.genotype_compression(soft_ops, hard_ops)
            return soft_ops, hard_ops
        else:
            used, possible = 0, 0
            for key, edge in self.edges.items():
                possible += edge.possible
                used += edge.used
            return used, possible

    def __repr__(self, out_format):
        dim_rep = self.in_dim[1:3]
        comp = self.genotype_compression(used_ratio=True)

        comp = comp[0]/comp[1] if comp[1] else np.nan
        layer_name = "Cell {:<2} {:<12}".format(self.name,'(' + self.cell_type + ")")
        dim = '{:^4}x{:^4}'.format(*dim_rep)
        out = out_format(l=layer_name, d=dim, p=general_num_params(self), c=comp)
        return out

    def detail_print(self, minimal):
        out = ""
        out += 'Size Est: {}\n'.format(sizeof_fmt(self.size_est*1024*1024))
        out += "X: {}, ({:,} params) \n".format(self.name-1 if self.name else 'In',
                                                general_num_params(self.scaler))
        out += "Y: {}, ({:,} params)\n".format(self.input_handler,
                                               general_num_params(self.input_handler))
        if not minimal:
            for key, edge in self.edges.items():
                out += '    {}\n'.format(edge)
        return out

    def get_parameters(self, selector):
        params = []
        for key, edge in self.edges.items():
            params += edge.get_parameters(selector)
        return params
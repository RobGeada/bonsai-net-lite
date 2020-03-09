import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__+"/../")))

import pickle as pkl
import pprint
import torch
import torch.nn as nn

from bonsai.data_loaders import load_data
from bonsai.helpers import mem_stats, clean, sizeof_fmt
from bonsai.net import Net
from bonsai.ops import commons, PrunableOperation
from bonsai.trainers import size_test



if __name__ == '__main__':
    if len(sys.argv)> 1 and sys.argv[1]=='o':
        # get sizes of individual operations in network
        # read args
        clean(verbose=False)
        C = int(sys.argv[3])
        dim = [int(x) for x in sys.argv[2:-1]]
        stride = int(sys.argv[-1])

        # build op
        op_mems = {}
        input_tensor = torch.zeros(dim,requires_grad=True).cuda()
        trials = 25

        for op, f in commons.items():
            sizes = []
            tensors = []
            for i in range(trials):
                if i==1:
                     start_mem = mem_stats(False)
                sm = mem_stats(False)
                op_f = PrunableOperation(f, op, mem_size=0, c_in=C, stride=stride).cuda()                
                out = op_f(input_tensor)
                tensors.append([op_f,out])
                sizes.append(sizeof_fmt(mem_stats(False) - sm))
            #print(op, sizes)
            end_mem = (mem_stats(False) - start_mem)/(trials-1)
            del tensors
            clean(verbose=False)
            op_mems[op] = end_mem / 1024 / 1024
        pp = pprint.PrettyPrinter(indent=0)
        pp.pprint(op_mems)
    else:
        # get size of entire model
        with open("pickles/size_test_in.pkl", "rb") as f:
            [n, e_c, add_pattern, prune, kwargs] = pkl.load(f)

        data, dim = load_data(kwargs['batch_size'], kwargs['dataset']['name'])
        model = Net(dim=dim,
                    classes=kwargs['dataset']['classes'],
                    scale=kwargs['scale'],
                    patterns=kwargs['patterns'],
                    num_patterns=n,
                    random_ops={'e_c': e_c, 'i_c': 1.},
                    nodes=kwargs['nodes'],
                    drop_prob=.3,
                    lr_schedule=kwargs['lr_schedule'],
                    prune=True)
        model.data = data

        if kwargs.get('remove_prune') is True:
            model.remove_pruners(remove_input=True, remove_edge=True)
            model.add_pattern(full_ops=True)
        elif add_pattern:
            model.add_pattern(prune=prune, random_ops={'e_c': add_pattern, 'i_c':100})
        if kwargs.get('detail', False):
            model.detail_print()
        if kwargs.get('print_model', False):
            print(model)
        out = size_test(model, verbose=kwargs.get('verbose', False))
        with open("pickles/size_test_out.pkl", "wb") as f:
            pkl.dump(out, f)

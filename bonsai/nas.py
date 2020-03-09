from bonsai.data_loaders import load_data
from bonsai.net import Net
from bonsai.trainers import *
from bonsai.helpers import *
from bonsai.ops import *


# === I/O ==============================================================================================================
def jn_print(x,end="\n"):
    print(x,end=end)
    with open("logs/jn_out.log", "a") as f:
        f.write(x+end)


# === INITIALIZATIONS ==================================================================================================
def gen_compression_targets(hypers):
    sizes = {}
    print("Search Range: {:.2f}->{:.2f}".format(1 / len(commons), 1.))
    for n in range(1, hypers['num_patterns']):
        sizes[n] = []
        bst = BST(1 / len(commons), 1.)
        while bst.answer is None:
            print("{}: {:.3f}\r".format(n, bst.pos), end="")
            queries = []
            for q in range(1):
                size = sp_size_test(n, e_c=bst.pos, add_pattern=1, remove_prune=False, **hypers)
                queries.append(not (not size[1] and (size[0]) < hypers['gpu_space']))
            bst.query(any(queries))
        if bst.passes:
            sizes[n] = max(bst.passes)

    if any([v for (k, v) in sizes.items() if v == 1]):
        start_size = [k for (k, v) in sizes.items() if v == 1][-1] + 1
    else:
        start_size = 1

    jn_print("Comp Ratios:\nsizes={{\n{}\n}}".format("\n".join(["    {}:{},".format(k, v) for (k, v) in sizes.items()])))
    jn_print("start_size={}".format(start_size))
    jn_print("Effective Scale: {:.2f}".format(hypers['scale']*sizes[max(sizes.keys())]*len(commons)))
    return sizes, start_size


# === MODEL CLASS ======================================================================================================
class Bonsai:
    def __init__(self, hypers, sizes=None, start_size=None):
        wipe_output()
        self.hypers = hypers
        self.id = namer()
        self.data, self.dim = load_data(hypers['batch_size'], hypers['dataset']['name'])
        self.hypers['num_patterns'] = get_n_patterns(hypers['patterns'],
                                                     self.dim,
                                                     hypers['reduction_target']) + 1

        if sizes is None:
            jn_print("== Determining compression ratios ==")
            self.sizes, self.start_size = gen_compression_targets(hypers)
        else:
            self.sizes, self.start_size = sizes, start_size
        self.model = None
        self.random = 0
        self.e_c, self.i_c = [], []
        
    # random levels:
    # 0: not random
    # 1: random at same level as penult. cell, after fully connected, prunable last cell
    # 2: random at same level as epoch 599, pruning
    # 3: random at same level as epoch 599, no pruning
    def generate_model(self):
        if self.random and self.e_c is None:
            raise ValueError("Cannot random search without e_c or i_c set")
        elif self.random:
            if self.random==1:
                random_ops = {'e_c': self.e_c[-2], 'i_c': self.i_c[-2]}
                prune = True
                num_patterns =  self.hypers['num_patterns']-1
            else:
                random_ops = {'e_c': self.e_c[-1], 'i_c': self.i_c[-1]}
                prune = self.random==2
                num_patterns = self.hypers['num_patterns']
            model_id = self.id + '_r{}'.format(self.random)
            jn_print("Generating model at random level {}, e_c={}, i_c={}, prune={}".format(
                    self.random,
                    random_ops['e_c'],
                    random_ops['i_c'],
                    prune))
        else:
            random_ops = None
            prune = True
            num_patterns = self.start_size
            
        self.model = Net(
            dim=self.dim,
            classes=self.hypers['dataset']['classes'],
            scale=self.hypers['scale'],
            patterns=self.hypers['patterns'],
            num_patterns=num_patterns,
            nodes=self.hypers['nodes'],
            random_ops=random_ops,
            prune=prune,
            model_id = model_id,
            drop_prob=self.hypers['drop_prob'],
            lr_schedule=self.hypers['lr_schedule'])
        
        if self.random==1:
            self.model.add_pattern()
        
        self.model.data = self.data

    def track_compression(self):
        _, e_c, i_c = self.model.genotype_compression()
        self.e_c.append(e_c)
        self.i_c.append(i_c)
        
    def train(self):
        if self.model is None:
            self.generate_model()

        if not self.random:
            # search
            search_start = time.time()
            for n in range(self.start_size, self.hypers['num_patterns']):
                jn_print(str(self.model))
                comp_ratio = self.sizes.get(n, 0)
                aim = comp_ratio * (.9 if self.sizes.get(n, 0) > .35 else .66)
                jn_print("=== {} Patterns. Target Comp: {:.2f}, Aim: {:.2f}".format(n, comp_ratio, aim))

                # learn+prune
                met_thresh = full_train(self.model,
                                        comp_lambdas=self.hypers['prune_rate'],
                                        comp_ratio=aim,
                                        size_thresh=self.sizes[n],
                                        nas_schedule=self.hypers['nas_schedule'])
                clean(verbose=False)
                if met_thresh:
                    if n != self.hypers['num_patterns']:
                        jn_print("Adding next pattern: {}".format(n + 1))
                        self.track_compression()
                        self.model.add_pattern()
                if not self.model.lr_schedulers['init'].remaining:
                    break

            # print search stats
            clean("Search End")
            jn_print("Search Time: {}".format(show_time(time.time() - search_start)))
            jn_print("Edge Comp: {} Input Comp: {}".format(self.e_c[-1], self.i_c[-1]))
            jn_print(str(self.model))
        else:
            jn_print(str(self.model))
            #self.model.detail_print()

        # train
        full_train(self.model,
                   epochs=self.model.lr_schedulers['init'].remaining,
                   nas_schedule=self.hypers['nas_schedule'])
        self.track_compression()
        clean()

    def random_search(self, level, e_c=None, i_c=None):
        self.random = level
        del self.model
        self.model = None
        if e_c is not None:
            self.e_c, self.i_c = [e_c]*2, [i_c]*2
        print(self.model)
        self.train()

import ast
import gc
import logging
import math
import numpy as np
import os
import pickle as pkl
import subprocess
import torch


# ==+ TORCH HELPERS ====================================================================================================
def clean(name=None, verbose=True):
    pre = mem_stats()
    gc.collect()
    torch.cuda.empty_cache()
    if verbose:
        print('Cleaning at {}. Pre: {}, Post: {}'.format(name, pre, mem_stats()))


def t_extract(x):
    return np.round(x.item(), 2)


def pretty_size(size):
    # pretty prints a torch.Size object
    assert (isinstance(size, torch.Size))
    return " x ".join(map(str, size)), int(np.prod(list(map(int, size))))


# Code by James Bradbury - https://github.com/jekbradbury
def print_obj_tree(min_elements=None):
    obj_list = [obj for obj in gc.get_objects() if torch.is_tensor(obj) or isinstance(obj, torch.autograd.Variable)]
    for obj in obj_list:
        if min_elements and obj.nelement() < min_elements: continue
        referrers = [r for r in gc.get_referrers(obj) if r is not obj_list]
        print(f'{id(obj)} {obj.__class__.__qualname__} of size {tuple(obj.size())} with references held by:')
        for referrer in referrers:
            if torch.is_tensor(referrer) or isinstance(referrer, torch.autograd.Variable):
                info_str = f' of size {tuple(referrer.size())}'
            elif isinstance(referrer, dict):
                info_str = ' in which its key is ', [k for k, v in referrer.items() if v is obj]
            else:
                info_str = ''
            print(f'  {id(referrer)} {referrer.__class__.__qualname__}{info_str}')


def size_obj(obj):
    print(sizeof_fmt(np.prod(obj.size()) * 32 * 8))


# === MODEL HELPERS ====================================================================================================
def schedule_generator(lr):
    return lambda x: {'lr_min': lr['lr_min'], 'lr_max': lr['lr_max'], 't_0': x, 't_mult': 1}


class LRScheduler:
    def __init__(self, T, lr_max):
        self.T = T
        self.lr_max = lr_max
        self.lr = lr_max
        self.remaining = T
        self.t = 0

    def step(self):
        self.t+=1
        self.remaining-=1
        self.lr = (.5 * self.lr_max) * (1 + np.cos((self.t * np.pi) / self.T))
        return self.lr
    
    def __repr__(self):
        return 'LRScheduler @ {:.2f}, {:.2f}->0 over {}/{} epochs'.format(self.lr, self.lr_max, self.t, self.T)

    
def get_n_patterns(patterns, dim, target=None):
    if target is None:
        target = np.log2(dim[2])
    tot_patterns,tot_reducts = 0, 0
    for i,pattern in enumerate(looping_generator(patterns)):
        reducts = 0
        for j,cell in enumerate(pattern):
            if 'r' in cell and not (i==0 and j==0):
                reducts += 1
        if tot_reducts+reducts<=target and (tot_reducts!=target):
            tot_reducts+=reducts
            tot_patterns+=1
        else:
            return tot_patterns
        

def cell_dims(data_shape, scale, patterns):
    size = list(data_shape)
    size[1] = scale
    sizes = set()
    for i, pattern in enumerate(patterns):
        for cell in pattern:
            if any(x == 0 for x in size):
                return sizes
            if i and 'r' in cell:
                sizes.add(tuple(size+[1]))
                size = list(channel_mod(size, size[1] * 2))
                if size[2]>1:
                    sizes.add(tuple(size+[2]))
                size = list(width_mod(size, 2))
            else:
                sizes.add(tuple(size+[1]))


def op_sizer(dims, single=True):
    if single:
        dims = [dims]
    sizes = {}
    for i,dim in enumerate(dims):
        print("\rSizing potential cell dim {} of {}: {}...".format(i+1,len(dims),dim),end="")
        try:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'sizer.py')
            python = get_python3()
            cmd = '{} {} o {}'.format(python, path, " ".join([str(x) for x in dim]))
            size = subprocess.check_output(cmd.split()).decode("ascii").strip()
            sizes[dim]=ast.literal_eval(size)
        except Exception as e:
            print(cmd)
            raise e
    print()
    return sizes


def compute_sizes(sizes=None):
    if sizes is not None:
        with open("pickles/op_sizes.pkl", "wb") as f:
            size_set = op_sizer(sizes, single=False)
            pkl.dump(size_set, f)
    else:
        size_set = {}
        if 'op_sizes.pkl' in os.listdir('pickles'):
            try:
                with open("pickles/op_sizes.pkl", "rb") as f:
                    size_set = pkl.load(f)
            except EOFError:
                pass
    return size_set


def general_num_params(model):
    # return number of differential parameters of input model
    return sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])


def channel_mod(dim, to):
    return dim[0], to, dim[2], dim[3]


def width_mod(dim, by):
    return dim[0], dim[1], dim[2] // by, dim[3] // by


def batch_mod(dim, by):
    return dim[0] // by, dim[1], dim[2], dim[3]


def cw_mod(dim, by):
    return dim[0], dim[1] * by, dim[2] // by, dim[3] // by


# === I/O HELPERS ======================================================================================================
class BST:
    def __init__(self, lower, upper, depth=6):
        self.lower = lower
        self.upper = upper
        self.step = (upper - lower) / 2 / 2
        self.pos = self.lower + (upper - lower) / 2
        self.depth = 0
        self.max_depth = depth
        self.min_step = (self.upper - self.lower) / (2 ** (self.max_depth - 1))
        self.answer = None
        self.passes = []

    def query(self, result):
        if self.pos <= self.lower:
            self.answer = self.lower
        elif self.pos >= self.upper:
            self.answer = self.upper
        elif self.depth == self.max_depth:
            self.answer = self.pos

        if result:
            self.pos -= self.step
            if self.step > self.min_step:
                self.step /= 2
            self.depth += 1
        else:
            self.passes.append(self.pos)
            self.pos += self.step
            if self.step > self.min_step:
                self.step /= 2
            self.depth += 1


def looping_generator(l):
    n = 0
    while 1:
        yield l[n % len(l)]
        n += 1


def div_remainder(n, interval):
    # finds divisor and remainder given some n/interval
    factor = math.floor(n / interval)
    remainder = int(n - (factor * interval))
    return factor, remainder


def namer():
    # generate random tripled-barrelled name to track models
    names = open("bonsai/names.txt", "r").readlines()
    len_names = len(names)
    choices = np.random.randint(0, len_names, 3)
    return " ".join([names[i].strip() for i in choices]).replace("'", "")


def sizeof_fmt(num, spacing=True, suffix='B'):
    # turns bytes object into human readable
    if spacing:
        fmt = "{:>7.2f}{:<3}" 
    else:
        fmt = "{:.2f}{}" 
        
    for unit in ['', 'Ki','Mi']:
        if abs(num) < 1024.0:
            return fmt.format(num, unit+suffix)
        num /= 1024.0
    return fmt.format(num, 'Gi'+suffix)


def cache_stats(human_readable=True, spacing=True):
    # returns current allocated torch memory
    if human_readable:
        return sizeof_fmt(torch.cuda.memory_cached(),spacing)
    else:
        return int(torch.cuda.memory_cached())


def mem_stats(human_readable=True, spacing=True):
    # returns current allocated torch memory
    if human_readable:
        return sizeof_fmt(torch.cuda.memory_allocated(),spacing)
    else:
        return int(torch.cuda.memory_allocated())

def mem_delta(start):
    return sizeof_fmt(mem_stats(False)-start)
    
    
def show_time(seconds):
    # show amount of time as human readable
    if seconds < 60:
        return "{:.2f}s".format(seconds)
    elif seconds < (60 * 60):
        minutes, seconds = div_remainder(seconds, 60)
        return "{}m,{}s".format(minutes, seconds)
    else:
        hours, seconds = div_remainder(seconds, 60 * 60)
        minutes, seconds = div_remainder(seconds, 60)
        return "{}h,{}m,{}s".format(hours, minutes, seconds)


def setup_logger(logger_name, filename, mode, terminator):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter()
    file_handler = logging.FileHandler(filename, mode=mode)
    file_handler.setFormatter(formatter)
    # file_handler.terminator = terminator

    l.setLevel(logging.INFO)
    l.addHandler(file_handler)


def log_print_curry(loggers):
    def log_print(string, end='\n', flush=False):
        print(string, end=end, flush=flush)
        for logger in loggers:
            if end == "\r":
                logger.info(string + "|carr_ret|")
            else:
                logger.info(string)

    return log_print


def prev_output(raw=False):
    with open("logs/jn_out.log", "r") as f:
        if raw:
            print(repr(f.read()))
        else:
            print(f.read().replace("|carr_ret|\n", "\r"))


def wipe_output():
    open("logs/jn_out.log", "w").close()


def get_python3():
    try:
        devnull = open(os.devnull)
        subprocess.Popen(["python3", "-V"], stdout=devnull, stderr=devnull).communicate()
    except OSError as e:
        print(e)
        if e.errno == os.errno.ENOENT:
            return 'python'
        else:
            raise e
    return 'python3'
from bonsai.nas import Bonsai

hypers = {
    'gpu_space':8.5,
    'dataset':{'name':'CIFAR10', 'classes':10},
    'batch_size':64,
    'scale':36,
    'nodes':4,
    'patterns': [['n','na'], ['r']],
    'reduction_target':2,
    'lr_schedule': {'lr_max': .01, 'T': 600},
    'drop_prob':.3,
    'nas_schedule': {'prune_interval':4, 'cycle_len':8},
    'prune_rate':{'edge':.01, 'input':.01}
}

bonsai = Bonsai(hypers)
bonsai.train()
bonsai.random_search(1)
bonsai.random_search(3)
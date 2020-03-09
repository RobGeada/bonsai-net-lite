import datetime
import time

import torch.nn as nn
import torch.optim as optim

from bonsai.helpers import *
from bonsai.ops import commons

# set up logging
setup_logger("training_logger", filename='logs/trainer.log', mode='a', terminator="\n")
setup_logger("jn_out", filename='logs/jn_out.log', mode='a', terminator="")
training_logger = logging.getLogger('training_logger')
jn_out = logging.getLogger('jn_out')

log_print = log_print_curry([training_logger, jn_out])
jn_print  = log_print_curry([jn_out])


# === EPOCH LEVEL FUNCTIONS ============================================================================================
def set_lr(optimizer, annealers):
    new_lrs = []
    for param_group in optimizer.param_groups:
        new_lr = annealers[param_group['key']].step() 
        param_group['lr']=new_lr
        new_lrs.append(new_lr)
    log_print("\n\x1b[31mAdjusting lrs to {}\x1b[0m".format(new_lrs))


def extract_curr_lambdas(comp_lambdas, epoch, tries):
    out = None
    if comp_lambdas:
        if epoch >= comp_lambdas['transition']:
            out = {k: v*2**tries for k, v in comp_lambdas['lambdas'].items()}
    return out


# === CUSTOM LOSS FUNCTIONS ============================================================================================
dummy_zero = torch.tensor([0.], device='cuda')


def compression_loss(model, comp_lambda, comp_ratio, item_output=False):
    # edge pruning
    if comp_lambda['edge']>0:
        prune_sizes = []
        for cell in model.cells:
            edge_pruners = [op.pruner.mem_size*op.pruner.sg() if op.pruner else dummy_zero\
                                for key, edge in cell.edges.items() for op in edge.ops]
            prune_sizes += [torch.sum(torch.cat(edge_pruners)).view(-1)]
        edge_comp_ratio = torch.div(torch.cat(prune_sizes), model.edge_sizes)
        edge_comp = torch.norm(comp_ratio - edge_comp_ratio)
        edge_loss = comp_lambda['edge'] * edge_comp
    else:
        edge_loss = 0

    # input pruning
    if comp_lambda['input']>0:
        input_sizes = []
        for cell in model.cells:
            input_pruners = [pruner.sg() if pruner else dummy_zero for pruner in cell.input_handler.pruners]
            input_sizes += [torch.sum(torch.cat(input_pruners)).view(-1)]
        input_comp_ratio = torch.div(torch.cat(input_sizes), model.input_p_tot)
        input_comp = torch.norm(1/model.input_p_tot - input_comp_ratio)
        input_loss = comp_lambda['input']*input_comp
    else:
        input_loss = 0

    loss = edge_loss+input_loss
    if item_output:
        ecr = 0 if edge_loss == 0 else torch.mean(edge_comp_ratio).item()
        icr = 0 if input_loss == 0 else torch.mean(input_comp_ratio).item()
        return loss, [ecr,icr], [edge_loss,input_loss]
    else:
        return loss, [None,None], [None,None]


# === PERFORMANCE METRICS ==============================================================================================
def top_k_accuracy(output, target, top_k, max_k):
    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].view(-1).float().sum(0) for k in top_k]


def accuracy_string(prefix, corrects, t_start, loader, top_k, comp_ratio=None, return_str=False):
    corrects = 100. * corrects / float(len(loader.dataset))
    out_string = "{} Corrects: ".format(prefix)
    for i, k in enumerate(top_k):
        out_string += 'Top-{}: {:.2f}%, '.format(k, corrects[i])
    if comp_ratio is not None:
        out_string += 'Comp: {:.2f}, {:.2f} '.format(*comp_ratio)
    out_string += show_time(time.time() - t_start)
    
    if return_str:
        return out_string
    else:
        log_print(out_string)


# === BASE LEVEL TRAIN AND TEST FUNCTIONS===============================================================================
def train(model, device, **kwargs):
    # === tracking stats ======================
    top_k = kwargs.get('top_k', [1])
    max_k = max(top_k)
    corrects = np.zeros(len(top_k), dtype=float)

    # === train epoch =========================
    model.train()
    train_loader = model.data[0]
    epoch_start = time.time()
    multiplier = kwargs.get('multiplier', 1)
    jn_print(datetime.datetime.now().strftime("%m/%d/%Y %I:%M %p"))

    t_data_start = None
    t_cumul_data, t_cumul_ops = 0,0
    for batch_idx, (data, target) in enumerate(train_loader):
        t_data_end = time.time()
        if t_data_start is not None:
            t_cumul_data += (t_data_end-t_data_start)
        t_op_start = time.time()

        print_or_end = (not batch_idx % 10) or (batch_idx == len(train_loader)-1)
        batch_start = time.time()

        # pass data ===========================
        data, target = data.to(device), target.to(device)
        if (batch_idx % multiplier == 0) or (batch_idx == len(train_loader)-1):
            kwargs['optimizer'].zero_grad()

        verbose = kwargs['epoch'] == 0 and batch_idx == 0
        outputs = model.forward(data, model.drop_prob, auxiliary=True, verbose=verbose)

        # classification loss =================
        def loss_f(x): return kwargs['criterion'](x, target)
        losses = [loss_f(output) for output in outputs[:-1]]
        final_loss = loss_f(outputs[-1])
        loss = final_loss + .2 * sum(losses)

        # compression loss ====================
        comp_ratio = None
        if kwargs.get('comp_lambda'):
            comp_loss, comp_ratio, loss_components = compression_loss(model,
                                                                      comp_lambda=kwargs.get('comp_lambda'),
                                                                      comp_ratio=kwargs['comp_ratio'],
                                                                      item_output = print_or_end)
            if print_or_end:
                loss_components = [loss] + loss_components
            loss += comp_loss

        # end train step ======================
        loss = loss/multiplier
        loss.backward()
        if (batch_idx % multiplier == 0) or (batch_idx == len(train_loader) - 1):
            kwargs['optimizer'].step()
        corrects = corrects + top_k_accuracy(outputs[-1], target, top_k=kwargs.get('top_k', [1]), max_k=max_k)

        # mid epoch updates ===================
        if print_or_end:
            losses_out = [np.round(x.item(), 2) for x in losses] + [np.round(final_loss.item(),2)]
            losses_out = ", ".join(["{}: {}".format(i,x) for i,x in enumerate(losses_out)])
            prog_str = 'Train Epoch: {:<3} [{:<6}/{:<6} ({:.0f}%)]\t'.format(
                kwargs['epoch'],
                (batch_idx + 1) * len(data),
                len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader))
            if kwargs.get('comp_lambda') is not None:
                prog_str += 'Comp Ratio: [E: {:.3f}, I: {:.3f}]'.format(*comp_ratio)
                prog_str += ', Loss Comp: [C: {:.3f}, E: {:.3f}, I: {:.2f}], '.format(*loss_components)
            prog_str += 'Per Epoch: {:<7}, '.format(show_time((time.time() - batch_start) * len(train_loader)))
            prog_str += 'Alloc: {}, '.format(cache_stats(spacing=False))
            prog_str += 'Data T: {:<6.3f}, Op T: {:<6.3f}'.format(t_cumul_data,t_cumul_ops)
            jn_print(prog_str, end="\r", flush=True)

        if batch_idx > kwargs.get('kill_at', np.inf):
            break
        t_cumul_ops += (time.time()-t_op_start)
        t_data_start = time.time()

    # === output ===============
    jn_print(prog_str)
    accuracy_string("Train", corrects, epoch_start, train_loader, top_k, comp_ratio=comp_ratio)
    if kwargs.get('comp_lambda') is not None:
        log_print("Train Loss Components: C: {:.3f}, E: {:.3"
                  "f}, I: {:.2f}".format(*loss_components))


def test(model, device, top_k=[1]):
    # === tracking stats =====================
    test_loader = model.data[1]
    max_k = max(top_k)
    corrects, e_corrects = np.zeros(len(top_k)), np.zeros(len(top_k))
    t_start = time.time()

    # === test epoch =========================
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            outputs = model.forward(data, drop_prob=0, auxiliary=True)
            #e_output = torch.mean(torch.stack(outputs, 1), 1)
            corrects = corrects + top_k_accuracy(outputs[-1], target, top_k=top_k, max_k=max_k)
            #e_corrects = e_corrects + top_k_accuracy(e_output, target, top_k=top_k, max_k=max_k)

    # === format results =====================
    #log_print(accuracy_string("All Towers Test ", e_corrects, t_start, test_loader, top_k, return_str=True))
    return accuracy_string("Last Tower Test ", corrects, t_start, test_loader, top_k, return_str=True)


def size_test(model, verbose=False):
    # run a few batches through the model to get an estimate of its GPU size
    try:
        start_size = cache_stats(False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for batch_idx, (data, target) in enumerate(model.data[0]):
            data, target = data.to(device), target.to(device)
            out = model.forward(data, auxiliary=True, verbose=(verbose and batch_idx==0))
            loss = criterion(out[-1], target)
            loss.backward()
            if batch_idx > 2:
                break
        overflow = False
        size = (cache_stats(False)-start_size)/1024/1024/1024
        clean(verbose=False)
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            overflow = True
            model = model.to(torch.device('cpu'))
            size = cache_stats(False)/(1024**3)
            clean(verbose=False)
            del model
            try:
                del data, target
            except:
                pass
            clean(verbose=False)
        else:
            raise e
    return size, overflow


def sp_size_test(n, e_c, add_pattern, prune=True,**kwargs):
    with open("pickles/size_test_in.pkl","wb") as f:
        pkl.dump([n, e_c, add_pattern, prune, kwargs],f)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sizer.py')
    python = get_python3()
    try:
        s=subprocess.check_output([python, path],stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print("Failed")
        print(python, path)
        print(e.output.decode('utf8'))
        raise e
    if kwargs.get('print_model',False):
        print(s.decode('utf8'))
    with open("pickles/size_test_out.pkl","rb") as f:
        return pkl.load(f)


# === FULL TRAINING HANDLER=============================================================================================
def full_train(model, epochs=None, **kwargs):

    # === learning handlers ==================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = [{'params':list(p),'lr':model.lr_schedulers[k].lr,'key':k} for k, p in model.pattern_params.items()]
    
    optimizer = optim.SGD(params, momentum=.9, weight_decay=3e-4)
    model.jn_print, model.log_print = jn_print, log_print
    print("=== Training {} ===".format(model.model_id))

    # === init logging =======================
    training_logger.info("=== NEW FULL TRAIN ===")
    log_print("Starting at {}".format(datetime.datetime.now()))
    training_logger.info(model.creation_string())

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

    # === run n epochs =======================
    if epochs is not None:
        epoch_ubound = model.lr_schedulers['init'].t+epochs
    else:
        epoch_ubound = model.lr_schedulers['init'].T
    epoch_lbound = model.lr_schedulers['init'].t
    met_thresh = False
    tries = 1
    for epoch in range(epoch_lbound, epoch_ubound):
        training_logger.info("=== EPOCH {} ===".format(epoch))

        # train =========================
        if kwargs.get('comp_lambdas') is not None:
            comp_lambda = {k: v*2**tries for k, v in kwargs['comp_lambdas'].items()}
        else:
            comp_lambda = None
        train(model, device, criterion=criterion, optimizer=optimizer, epoch=epoch, comp_lambda=comp_lambda, **kwargs)
        model.save_genotype()

        # prune ==============================
        if model.prune['edge']:
            model.eval()
            edge_pruners = [op.pruner for cell in model.cells \
                            for key, edge in cell.edges.items() for op in edge.ops if op.pruner]
            input_pruners = [pruner for cell in model.cells \
                            for pruner in cell.input_handler.pruners if pruner]
            [pruner.track_gates() for pruner in edge_pruners + input_pruners]

            if epoch and not (epoch + 1) % kwargs['nas_schedule']['prune_interval']:
                model.deadhead()

            hard, soft = model.genotype_compression()[:2]
            if soft and hard:
                jn_print("Soft Comp: {:.3f}, Hard Comp: {:.3f}".format(soft, hard))
            if epochs is None and epoch >= (epoch_lbound + (kwargs['nas_schedule']['cycle_len'] * (tries))):
                jn_print("Starting compression cycle {}...".format(tries+1))
                tries += 1

            if hard < kwargs.get('size_thresh', 1):
                met_thresh = True

        # test ===============================
        log_print(test(model, device, top_k=kwargs.get('top_k', [1])))

        # anneal =============================
        set_lr(optimizer, model.lr_schedulers)

        if met_thresh and epochs is None:
            break 
    return met_thresh

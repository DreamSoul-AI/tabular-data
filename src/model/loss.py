import torch
import torch.nn.functional as F


def make_loss(output, target, **args):
    loss = loss_fn(output, target, **args)
    return loss


def loss_fn(output, target, reduction='mean', mode='classification', log_prob=False):
    if mode == 'classification':
        if target.dtype == torch.int64:
            if output.dtype == torch.int64:
                loss = output.eq(target).float().mean()
            else:
                if log_prob:
                    loss = F.nll_loss(output, target, reduction=reduction)
                else:
                    loss = F.cross_entropy(output, target, reduction=reduction)
        else:
            loss = kld_loss(output, target, reduction=reduction)
    elif mode == 'regression':
        loss = F.mse_loss(output, target, reduction=reduction)
    else:
        raise ValueError('Not valid mode')
    return loss


def cross_entropy_loss(output, target, reduction='mean'):
    if target.dtype != torch.int64:
        target = (target.topk(1, 1, True, True)[1]).view(-1)
    ce = F.cross_entropy(output, target, reduction=reduction)
    return ce


def kld_loss(output, target, reduction='batchmean'):
    kld = F.kl_div(F.log_softmax(output, dim=-1), target, reduction=reduction)
    return kld

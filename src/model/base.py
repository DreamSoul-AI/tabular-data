import torch
import torch.nn as nn
from .model import make_loss, normalize


class Base(nn.Module):
    def __init__(self, core, stats, task_mode):
        super().__init__()
        self.core = core
        self.task_mode = task_mode
        self.register_buffer('data_mean', stats['data'].mean)
        self.register_buffer('data_std', stats['data'].std)
        if task_mode == 'regression':
            self.register_buffer('target_mean', stats['target'].mean)
            self.register_buffer('target_std', stats['target'].mean)

    def forward(self, input):
        output = {}
        x = input['data']
        x = normalize(x, 1 / self.data_std, -self.data_mean / self.data_std)
        x = self.core.f(x)
        output['target'] = x
        if self.task_mode == 'regression':
            input['target'] = normalize(input['target'], 1 / self.target_std, -self.target_mean / self.target_std)
        output['loss'] = make_loss(output, input, mode=self.task_mode)
        if self.task_mode == 'regression':
            output['target'] = normalize(output['target'], self.target_std, self.target_mean)
            input['target'] = normalize(input['target'], self.target_std, self.target_mean)
        return output


def base(base, stats, task_mode):
    model = Base(base, stats, task_mode)
    return model

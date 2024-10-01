import torch
import torch.nn as nn
from .model import normalize
from .loss import make_loss


class Base(nn.Module):
    def __init__(self, core, stats, task_mode, index):
        super().__init__()
        self.core = core
        self.index = index
        self.task_mode = task_mode
        self.register_buffer('data_mean', stats[index]['data'].mean)
        self.register_buffer('data_std', stats[index]['data'].std)
        if task_mode == 'regression':
            self.register_buffer('target_mean', stats[index]['target'].mean)
            self.register_buffer('target_std', stats[index]['target'].mean)

    def forward(self, input):
        output = {}
        x = self.normalize_input(input)
        x = self.core(x)
        output['pred'] = x
        output['loss'] = make_loss(output, input, mode=self.task_mode)
        self.normalize_output(input, output)
        return output

    def normalize_input(self, input):
        x = input['data']
        x = normalize(x, 1 / self.data_std, -self.data_mean / self.data_std)
        if self.task_mode == 'regression':
            input['target'] = normalize(input['target'], 1 / self.target_std, -self.target_mean / self.target_std)
        return x

    def normalize_output(self, input, output):
        if self.task_mode == 'regression':
            output['pred'] = normalize(output['pred'], self.target_std, self.target_mean)
            input['target'] = normalize(input['target'], self.target_std, self.target_mean)
        return


def base(core, cfg, index):
    stats = cfg['stats']
    task_mode = cfg['task_mode']
    model = Base(core, stats, task_mode, index)
    return model

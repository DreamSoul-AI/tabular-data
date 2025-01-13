import torch
import torch.nn as nn
from .model import normalize
from .loss import make_loss


class Numeric(nn.Module):
    def __init__(self, core, cfg, index):
        super().__init__()
        self.core = core
        self.index = index
        self.cfg = cfg
        self.data_mode = cfg['data_mode']
        self.model_name = cfg['model_name']
        self.task_mode = cfg['task_mode']
        self.stats = cfg['stats']
        self.register_buffer('data_mean', self.stats[index]['numeric']['data'].mean)
        self.register_buffer('data_std', self.stats[index]['numeric']['data'].std)
        if self.task_mode == 'regression':
            self.register_buffer('target_mean', self.stats[index]['numeric']['target'].mean)
            self.register_buffer('target_std', self.stats[index]['numeric']['target'].mean)

    def forward(self, input):
        output = {}
        x = self.make_input(input)
        x = self.core(x)
        output['pred'] = x
        if 'target' in input:
            output['loss'] = make_loss(output['pred'], input['target'], mode=self.task_mode)
        self.make_output(input, output)
        return output

    def make_input(self, input):
        x = input['numeric']['data']
        x = normalize(x, 1 / self.data_std, -self.data_mean / self.data_std)
        if 'target' in input['numeric']:
            input['target'] = input['numeric']['target']

        if self.task_mode == 'regression':
            if 'target' in input['numeric']:
                input['target'] = normalize(input['target'], 1 / self.target_std,
                                            -self.target_mean / self.target_std)
        return x

    def make_output(self, input, output):
        if self.task_mode == 'regression':
            output['pred'] = normalize(output['pred'], self.target_std, self.target_mean)
            if 'target' in input:
                input['target'] = normalize(input['target'], self.target_std, self.target_mean)
        return


def numeric(core, cfg, index):
    model = Numeric(core, cfg, index)
    return model

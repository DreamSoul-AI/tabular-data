import torch
import torch.nn as nn
from .model import normalize
from .loss import make_loss


class Base(nn.Module):
    def __init__(self, core, index, cfg):
        super().__init__()
        self.core = core
        self.index = index
        self.cfg = cfg
        self.model_name = cfg['model_name']
        self.task_mode = cfg['task_mode']
        self.stats = cfg['stats']
        self.register_buffer('data_mean', self.stats[index]['data'].mean)
        self.register_buffer('data_std', self.stats[index]['data'].std)
        if self.task_mode == 'regression':
            self.register_buffer('target_mean', self.stats[index]['target'].mean)
            self.register_buffer('target_std', self.stats[index]['target'].mean)

    def forward(self, input):
        output = {}
        x = self.normalize_input(input)
        x = self.core(x)
        output['pred'] = x
        output['loss'] = make_loss(output, input, mode=self.task_mode)
        self.normalize_output(input, output)
        # https://huggingface.co/dunzhang/stella_en_400M_v5
        return output

    def normalize_input(self, input):
        x = input['data']
        x = normalize(x, 1 / self.data_std, -self.data_mean / self.data_std)
        if self.task_mode == 'regression':
            if 'target' in input:
                input['target'] = normalize(input['target'], 1 / self.target_std, -self.target_mean / self.target_std)
        return x

    def normalize_output(self, input, output):
        if self.task_mode == 'regression':
            output['pred'] = normalize(output['pred'], self.target_std, self.target_mean)
            if 'target' in input:
                input['target'] = normalize(input['target'], self.target_std, self.target_mean)
        return


def base(core, cfg, index):
    model = Base(core, index, cfg)
    return model

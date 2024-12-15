import torch
import torch.nn as nn
from .model import normalize
from .loss import make_loss


class Semantic(nn.Module):
    def __init__(self, core, tokenizer, index, cfg):
        super().__init__()
        self.core = core
        self.tokenizer = tokenizer
        self.index = index
        self.cfg = cfg
        self.data_mode = cfg['data_mode']
        self.model_name = cfg['model_name']
        self.task_mode = cfg['task_mode']

    def forward(self, input):
        output = {}
        input = self.make_mask(input)
        input = self.flatten(input)
        x = self.core(**input)
        print(x.last_hidden_state.size())
        exit()
        output['pred'] = x
        output['loss'] = make_loss(output, input, mode=self.task_mode, log_prob=True)
        self.normalize_output(input, output)
        return output

    def make_mask(self, input):
        if self.cfg['mask_mode'] == 'target':
            input['input_ids'][-1][:] = self.tokenizer.mask_token_id
            input['attention_mask'][-1][:] = 1
        return input

    def flatten(self, input):
        input['input_ids'] = input['input_ids'].view(input['input_ids'].size(0), -1)
        input['attention_mask'] = input['attention_mask'].view(input['attention_mask'].size(0), -1)
        return input


def semantic(core, tokenizer, cfg, index):
    model = Semantic(core, tokenizer, index, cfg)
    return model

import torch
import torch.nn as nn
from .model import normalize
from .loss import make_loss


class Semantic(nn.Module):
    def __init__(self, core, tokenizer, index, cfg):
        super().__init__()
        self.core = core
        self.freeze(self.core)
        self.tokenizer = tokenizer
        self.index = index
        self.cfg = cfg
        self.data_mode = cfg['data_mode']
        self.task_mode = 'classification'
        self.out_proj = nn.Linear(cfg[cfg['model_name']]['hidden_size'], tokenizer.vocab_size) # TODO: use bert default out proj

    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return

    def forward(self, input):
        output = {}
        input, target = self.make_target(input)
        input = self.flatten(input)
        x = self.core(**input)
        pred = self.make_pred(x.last_hidden_state)
        output['pred'] = pred
        input['target'] = target
        output['loss'] = make_loss(output, input, mode=self.task_mode, log_prob=False)
        return output

    def make_target(self, input):
        if self.cfg['mask_mode'] == 'target':
            target = input['input_ids'][:, -1].clone().detach()
            input['input_ids'][:, -1][:] = self.tokenizer.mask_token_id
            input['attention_mask'][:, -1][:] = 1
        return input, target

    def flatten(self, input):
        input['input_ids'] = input['input_ids'].view(input['input_ids'].size(0), -1)
        input['attention_mask'] = input['attention_mask'].view(input['attention_mask'].size(0), -1)
        return input

    def make_pred(self, hidden_state):
        if self.cfg['mask_mode'] == 'target':
            hidden_state = hidden_state.view(hidden_state.size(0), -1, self.cfg['max_length'], hidden_state.size(-1))
            hidden_state = hidden_state[:, -1]
        pred = self.out_proj(hidden_state).transpose(1, 2)
        return pred


def semantic(core, tokenizer, cfg, index):
    model = Semantic(core, tokenizer, index, cfg)
    return model

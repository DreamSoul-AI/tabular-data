import torch
import torch.nn as nn
from .model import normalize
from .loss import make_loss
from module import filter_args


class Semantic(nn.Module):
    def __init__(self, core, tokenizer, cfg, index):
        super().__init__()
        self.core = core
        self.freeze(self.core)
        self.tokenizer = tokenizer
        self.index = index
        self.cfg = cfg
        self.task_mode = 'classification'
        self.hidden_size = cfg[cfg['model_name']]['hidden_size']
        # self.adapter = nn.Linear(self.hidden_size, self.hidden_size)
        # self.out_proj = nn.Linear(self.hidden_size, tokenizer.vocab_size, bias=False)
        # for param_name, param in self.core.named_parameters():
        #     if 'word_embeddings' in param_name:
        #         self.out_proj.weight = param
        #         self.out_proj.weight.requires_grad = False
        #         break
        self.out_proj = nn.Linear(self.hidden_size, tokenizer.vocab_size)

    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return

    def forward(self, input):
        output = {}
        input, target = self.make_target(input)
        input = self.flatten(input)
        valid_input = filter_args(self.core.forward, input)
        x = self.core(**valid_input)
        x = x.last_hidden_state
        # x = self.adapter(x)
        output['pred'], output['pred_semantic'], output['pred_numeric'] = self.make_pred(x)
        input['target'] = target
        output['loss'] = make_loss(output, input, mode=self.task_mode, log_prob=False)
        input['target'] = input['target_numeric']
        output['pred'] = output['pred_numeric']
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
            hidden_state = torch.cumsum(hidden_state, dim=1) # pos embedding
            # print(hidden_state.size())
            # exit()
        pred = self.out_proj(hidden_state)
        pred = pred.transpose(1, 2)
        # print(pred.size())
        pred_tokens = torch.argmax(pred, dim=1)
        print(pred_tokens)
        # print(pred_tokens.size())
        pred_semantic = self.tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
        pred_numeric = [self.classes_to_labels.get(pred_semantic[i], -1) for i in range(len(pred_semantic))]
        pred_numeric = hidden_state.new_tensor(pred_numeric, dtype=torch.long)
        print(pred_semantic)
        return pred, pred_semantic, pred_numeric


def semantic(core, tokenizer, cfg, index):
    model = Semantic(core, tokenizer, cfg, index)
    return model

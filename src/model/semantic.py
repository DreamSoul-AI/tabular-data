import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import normalize
from .loss import make_loss
from module import filter_args


class RopePositionEmbedding(torch.nn.Module):
    def __init__(self, dim, max_len=512):
        """
        Args:
            dim (int): The dimension of the embeddings (e.g., 512).
            max_len (int): Maximum length of the input sequence.
        """
        super(RopePositionEmbedding, self).__init__()
        self.dim = dim
        self.max_len = max_len

        # Create the position tensor (0 to max_len - 1)
        positions = torch.arange(self.max_len, dtype=torch.float32).unsqueeze(1)
        self.register_buffer('positions', positions)

        # Create frequency terms
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        """
        Args:
            x (Tensor): The input tensor (batch_size, seq_len, dim)
        """
        seq_len = x.size(1)

        # Calculate the position embeddings (seq_len, dim) using rotation
        pos_emb = torch.matmul(self.positions[:seq_len], self.inv_freq.unsqueeze(0))

        # Apply rotation for odd and even dimensions
        pos_emb = pos_emb.unsqueeze(0)  # (1, seq_len, dim//2)
        sin_pos_emb = torch.sin(pos_emb)
        cos_pos_emb = torch.cos(pos_emb)

        # Concatenate sin and cos along the last dimension
        pos_emb = torch.cat([sin_pos_emb, cos_pos_emb], dim=-1)

        # Broadcast the positional encoding to the input tensor shape
        return x + pos_emb


class Semantic(nn.Module):
    def __init__(self, core, tokenizer, cfg, index):
        super().__init__()
        self.core = core
        self.freeze(self.core)
        self.tokenizer = tokenizer
        self.index = index
        self.cfg = cfg
        self.task_mode = 'classification'
        self.hidden_size = core.config.hidden_size
        self.target_size = tokenizer.vocab_size
        # self.pos_embedding = RopePositionEmbedding(self.hidden_size, max_len=cfg['max_length'])
        self.out_proj = nn.Linear(self.hidden_size, self.target_size)

    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return

    def forward(self, input):
        output = {}
        self.make_attention_mask(input['semantic'])

        # input, target, target_weight = self.make_target(input['semantic'])
        input, sequence_length = self.flatten(input)
        valid_input = filter_args(self.core.forward, input)
        with torch.no_grad():
            x = self.core(**valid_input)
        x = x.last_hidden_state.detach()
        output['pred'], output['pred_semantic'], output['pred_numeric'] = self.make_pred(x, sequence_length)
        input['target'] = target
        # print(input['target'])
        # exit()
        # output['loss'] = make_loss(output, input, mode=self.task_mode, log_prob=False)
        output['loss'] = F.cross_entropy(output['pred'], input['target'], reduction='mean', weight=target_weight)
        input['target'] = input['target_numeric']
        output['pred'] = output['pred_numeric']
        return output

    def make_attention_mask(self, input):
        ## TODO: size is [256, 34, 8], need generate attention mask so that the last token is not attended
        ## and the second last token output should be directed to output the last token (the column name is the query)
        ## use the word embeddings only from the bert model and then write another attention module with the new mask
        ## or use the new mask in the bert model encoder
        ## need a projection layer to project 8 * bert_embedding_size to embedding
        # print(input['input_ids'].size(), input['attention_mask'].size(), input['attention_mask'].size())
        S = input['input_ids'].size(1)
        # S = 8
        mask = input['input_ids'].new_zeros((S, S), dtype=torch.float)
        n = S // 2
        for i in range(n):
            # Feature name f_i can attend to all positions except v_i
            mask[2 * i, :] = 0
            # f_i cannot attend to its own value v_i
            mask[2 * i, 2 * i + 1] = -float('inf')
            mask[2 * i + 1, :] = -float('inf')
            mask[2 * i + 1, 2 * i + 1] = 0
        print(input['attention_mask']) # TODO: merge with the above mask for padding -> (34, 34, 8)
        print(mask)
        exit()
        # For each value v_i, allow it to attend only to its corresponding feature f_i
        for i in range(n):
            mask[2 * i + 1, 2 * i] = 1  # v_i can only attend to f_i

        exit()
        # target = input['input_ids'][:, -1].clone().detach()
        # target_semantic = self.tokenizer.batch_decode(target, skip_special_tokens=True)
        # target_weight = target.new_zeros((self.target_size,), dtype=torch.float)
        # unique_target, unique_counts = torch.unique(target, return_counts=True)
        # unique_freq = 1 / unique_counts
        # target_weight[unique_target] = unique_freq / unique_freq.sum()
        # input['input_ids'][:, -1][:] = self.tokenizer.mask_token_id
        # input['attention_mask'][:, -1][:] = 1
        return input, target, target_weight

    def flatten(self, input):
        sequence_length = input['input_ids'].size(1)
        input['input_ids'] = input['input_ids'].view(-1, input['input_ids'].size(-1))
        input['attention_mask'] = input['attention_mask'].view(-1, input['attention_mask'].size(-1))
        return input, sequence_length

    def make_pred(self, hidden_state, sequence_length):
        if self.cfg['mask_mode'] == 'target':
            hidden_state = hidden_state.view(-1, sequence_length, self.cfg['max_length'], hidden_state.size(-1))
            # hidden_state = hidden_state[:, -1]
            hidden_state = hidden_state.mean(dim=1)
            hidden_state = self.pos_embedding(hidden_state)
            # hidden_state = torch.cumsum(hidden_state, dim=1)  # pos embedding
        pred = self.out_proj(hidden_state)
        pred = pred.transpose(1, 2)
        # print(pred.size())
        pred_tokens = torch.argmax(pred, dim=1)
        # print(pred_tokens)
        # print(pred_tokens.size())
        pred_semantic = self.tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
        pred_numeric = [self.classes_to_labels.get(pred_semantic[i], -1) for i in range(len(pred_semantic))]
        pred_numeric = hidden_state.new_tensor(pred_numeric, dtype=torch.long)
        print('pred', pred_semantic)
        return pred, pred_semantic, pred_numeric


def semantic(core, tokenizer, cfg, index):
    model = Semantic(core, tokenizer, cfg, index)
    return model

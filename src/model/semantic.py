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
        self.encode_length = cfg['encode_length']
        self.hidden_size = cfg['bert']['hidden_size']
        self.target_size = tokenizer.vocab_size
        # TODO: use data embedding
        self.in_proj = nn.Linear(self.core.config.hidden_size * self.encode_length, self.hidden_size)
        # self.pos_embedding = RopePositionEmbedding(self.hidden_size, max_len=cfg['max_length'])
        self.out_proj = nn.Linear(self.hidden_size, self.target_size)

    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return

    def encode(self, input):
        sequence_length = input['input_ids'].size(1)
        input_ids = input['input_ids'].view(-1, input['input_ids'].size(-1))
        attention_mask = input['attention_mask'].view(-1, input['attention_mask'].size(-1))
        encoder_input = {'input_ids': input_ids, 'attention_mask': attention_mask}
        with torch.no_grad():
            x = self.core(**encoder_input)
            x = x.last_hidden_state.detach()
        x = x.view(-1, sequence_length, *x.shape[1:])
        x = x.view(*x.shape[:2], -1)
        x = self.in_proj(x)
        return x

    def forward(self, input):
        output = {}
        feature_mask = self.make_feature_mask(input['semantic'])
        x = self.encode(input['semantic'])
        output['pred'], output['pred_semantic'], output['pred_numeric'] = self.make_pred(x, sequence_length)
        input['target'] = target
        # print(input['target'])
        # exit()
        # output['loss'] = make_loss(output, input, mode=self.task_mode, log_prob=False)
        output['loss'] = F.cross_entropy(output['pred'], input['target'], reduction='mean', weight=target_weight)
        input['target'] = input['target_numeric']
        output['pred'] = output['pred_numeric']
        return output

    def make_feature_mask(self, input):
        ## TODO: size is [256, 34, 8], need generate attention mask so that the last token is not attended
        ## and the second last token output should be directed to output the last token (the column name is the query)
        ## use the word embeddings only from the bert model and then write another attention module with the new mask
        ## or use the new mask in the bert model encoder
        ## need a projection layer to project 8 * bert_embedding_size to embedding
        # print(input['input_ids'].size(), input['attention_mask'].size(), input['attention_mask'].size())
        sequence_length = input['input_ids'].size(1)
        # S = 8
        feature_mask = input['input_ids'].new_zeros((sequence_length, sequence_length), dtype=torch.long)
        n = sequence_length // 2
        for i in range(n):
            # Feature name f_i can attend to all positions except v_i
            feature_mask[2 * i, :] = 1
            # f_i cannot attend to its own value v_i
            feature_mask[2 * i, 2 * i + 1] = 0
            feature_mask[2 * i + 1, :] = 0
            feature_mask[2 * i + 1, 2 * i + 1] = 1
        # feature_mask = feature_mask.unsqueeze(-1)
        # attention_mask = input['attention_mask'].unsqueeze(1)
        # mask = torch.logical_and(feature_mask, attention_mask)
        # exit()
        # TODO: input_ids (256, 34, 8) -> (256, 34, 8, D) -> (256, 34, 8 * D) -> (256, 34, 128) [N, S, D] use pretrained or customized embedding
        # attention (256, 34, 34) -> attention-based model -> (256, 34, 128) -> (256, 34, 8 * D)- > (256, 34, 8, D) -> (256, 34, 8) -> decode
        # return input, target, target_weight
        return feature_mask

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

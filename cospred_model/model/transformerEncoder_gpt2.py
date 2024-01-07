'''
    code from TaeHwan Jung(@graykode), modified by Liang Xue
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import copy

from cospred_model.metrics import spectral_distance, masked_spectral_distance

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
    
class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present

# class SelfAttention(nn.Module):
#     """
#     A vanilla multi-head masked self-attention layer with a projection at the end.
#     It is possible to use torch.nn.MultiheadAttention here but I am including an
#     explicit implementation here to show that there is nothing too scary here.
#     """

#     def __init__(self, config):
#         super().__init__()
#         assert config.n_embd % config.n_head == 0
#         # key, query, value projections for all heads
#         self.key = nn.Linear(config.n_embd, config.n_embd)
#         self.query = nn.Linear(config.n_embd, config.n_embd)
#         self.value = nn.Linear(config.n_embd, config.n_embd)
#         # regularization
#         self.attn_drop = nn.Dropout(config.attn_pdrop)
#         self.resid_drop = nn.Dropout(config.resid_pdrop)
#         # output projection
#         self.proj = nn.Linear(config.n_embd, config.n_embd)
#         # causal mask to ensure that attention is only applied to the left in the input sequence
#         self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
#                              .view(1, 1, config.block_size, config.block_size))
#         self.n_head = config.n_head

#     def forward(self, x, layer_past=None):
#         B, T, C = x.size()

#         # calculate query, key, values for all heads in batch and move head forward to be the batch dim
#         k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
#         q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
#         v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

#         # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
#         att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
#         # att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
#         att = F.softmax(att, dim=-1)
#         att = self.attn_drop(att)
#         y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
#         y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

#         # output projection
#         y = self.resid_drop(self.proj(y))
#         return y


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2

class Block(nn.Module):
    def __init__(self, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, config.n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present


class BlockList(nn.Module):
    def __init__(self, block, config):
        super().__init__()
        self.blocks = nn.ModuleList([
            copy.deepcopy(block) for _ in range(config.n_layer)
        ])

    def forward(self, x):
        for block in self.blocks:
            x, _ = block(x)
        return x
    
    def __iter__(self):
        return iter(self.blocks)
    
# class Block(nn.Module):
#     """ an unassuming Transformer block """
#     def __init__(self, config):
#         super().__init__()
#         self.ln1 = nn.LayerNorm(config.n_embd)
#         self.attn = SelfAttention(config)
#         self.ln2 = nn.LayerNorm(config.n_embd)
#         self.mlp = nn.Sequential(
#             nn.Linear(config.n_embd, 4 * config.n_embd),
#             nn.GELU(),
#             nn.Linear(4 * config.n_embd, config.n_embd),
#             nn.Dropout(config.resid_pdrop)
#         )

#     def forward(self, x):
#         x = x + self.attn(self.ln1(x))
#         x = x + self.mlp(self.ln2(x))
#         return x



class GPT2LMHead(nn.Module):
    def __init__(self, model_embeddings_weights, config):
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        lm_logits = self.decoder(hidden_state)
        return lm_logits

class GPT2LMHeadModel(nn.Module):
    def __init__(self, config):
        super(GPT2LMHeadModel, self).__init__()
        self.transformer = GPT2Model(config)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)

    def set_tied(self):
        """ Make sure we are sharing the embeddings
        """
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None, past=None):
        hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past)
        lm_logits = self.lm_head(hidden_states)
        if lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            return loss
        return lm_logits, presents
    

# class TransformerConfig:
#     """ base GPT config, params common to all GPT versions """
#     embd_pdrop = 0.1
#     resid_pdrop = 0.1
#     attn_pdrop = 0.1

#     def __init__(self, vocab_size, block_size, **kwargs):
#         self.vocab_size = vocab_size
#         self.block_size = block_size
#         for k, v in kwargs.items():
#             setattr(self, k, v)

'''
    code by TaeHwan Jung(@graykode), modified by Liang Xue
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
class TransformerConfig(object):
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(
            self,
            vocab_size,
            block_size, 
            n_positions=1024,
            n_ctx=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            **kwargs
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        for k, v in kwargs.items():
            setattr(self, k, v)


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.charge_emb = nn.Embedding(config.max_charge, config.n_embd)
        self.ce_emb = nn.Embedding(config.max_ce, config.n_embd)

        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        
        # transformer
        # self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        block = Block(config, scale=True)
        # self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.blocks = BlockList(block, config)
        
        # decoder head
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        # self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.n_output, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, Conv1D, MLP)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, LayerNorm)
        for mn, m in self.named_modules():
            # print('mn={}, m={}'.format(mn, m))
            for pn, p in m.named_parameters():
                # print('pn={}, p={}'.format(pn, p))
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                # print('fpn={}'.format(fpn)) 
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate,
                                      betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None):
        b, t = idx.size()
        idx = idx.long()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the Encoder model
        token_embeddings = self.tok_emb(idx[:, range(30)])  # each index maps to a (learnable) vector
        charge_embeddings = self.charge_emb(idx[:, range(30,36)])
        ce_embeddings = self.ce_emb(idx[:, [36]])
        # token_embeddings = self.tok_emb(idx[:, 1:])  # each index maps to a (learnable) vector
        # charge_embeddings = self.charge_emb(idx[:, [0]])
        token_embeddings = torch.cat([charge_embeddings, token_embeddings, ce_embeddings], dim=1)
        # print(token_embeddings.shape)

        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        # for block in self.blocks:
        #     x = block(x)
        x = self.blocks(x)
        x = self.ln_f(x)

        x_sum = x.sum(dim=1)
        # x_max, _ = x.max(dim=1)
        outputs = self.head(x_sum)
        # print("outputs")
        # print(outputs.shape)
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            # loss = torch.mean((outputs-targets)**2)
            loss = spectral_distance(outputs, targets)
            # loss = masked_spectral_distance(outputs, targets)
            # loss = F.mse_loss(outputs, targets)
            # print(loss)
        return outputs, loss
    

    # def forward(self, input_ids, position_ids=None, token_type_ids=None, past=None):
    #     if past is None:
    #         past_length = 0
    #         past = [None] * len(self.h)
    #     else:
    #         past_length = past[0][0].size(-2)
    #     if position_ids is None:
    #         position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long,
    #                                     device=input_ids.device)
    #         position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

    #     input_shape = input_ids.size()
    #     input_ids = input_ids.view(-1, input_ids.size(-1))
    #     position_ids = position_ids.view(-1, position_ids.size(-1))

    #     inputs_embeds = self.wte(input_ids)
    #     position_embeds = self.wpe(position_ids)
    #     if token_type_ids is not None:
    #         token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
    #         token_type_embeds = self.wte(token_type_ids)
    #     else:
    #         token_type_embeds = 0
    #     hidden_states = inputs_embeds + position_embeds + token_type_embeds
    #     presents = []
    #     for block, layer_past in zip(self.h, past):
    #         hidden_states, present = block(hidden_states, layer_past)
    #         presents.append(present)
    #     hidden_states = self.ln_f(hidden_states)
    #     output_shape = input_shape + (hidden_states.size(-1),)
    #     return hidden_states.view(*output_shape), presents
    


    
    
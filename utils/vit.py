import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from utils.pe import PositionalEncoding, ScaledPositionalEncoding

# helpers

def get_activation_fn(activation):
    activation = activation.strip().lower()
    assert activation in ("relu", "prelu", "gelu",)
    if activation=="relu":
        return nn.ReLU()
    elif activation=="prelu":
        return nn.PReLU()
    else:
        return nn.GELU()

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., activation="gelu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., activation="gelu"):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout, activation=activation))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class TransformerEncoder(nn.Module):  # the transformer encoder modified for temporal data

    def __init__(
            self, *,
            num_classes=1,  # number of classes to predict
            dim,   # dimension of the input embedding
            depth,   # depth of transformer
            heads,   # head number of transformer
            ff_dim,  # MLP dimension of transformer's feedforward layer
            mlp_head_hidden_dim=None, # the hidden layer dimensions of the MLP head
            use_class_token=True,   # whether to use the class token or not
            pool = 'cls',  #  how to get the final prediction, 'cls': use class token, 'mean': mean pooling of the sequence
            dim_head = 64,  # head dimension of transformer's attention module
            dropout = 0.,   # dropout rate of transformer
            mlp_head_dropout = 0.,  # dropout rate of the MLP head
            pe_method=None, # positional embedding method; None->disable PE; "origin"->original PE; "scale": scaled PE
            pe_max_len=5000,   # the maximum length of all input sequences (required by positional embedding)
            activation="gelu",   # the activation function, can be "gelu", "relu" or "prelu"

    ):

        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        assert pe_method in (None, "origin", "scale",), \
            "Invalid pe method: {}".format(self.positional_embedding_method)

        if pe_method == "origin":
            # original PE used in Attention is All You Need
            self.pos_embedding = PositionalEncoding(
                d_model=dim, dropout_rate=dropout, max_len=pe_max_len)
        elif pe_method == "scale":
            # scaled PE (https://arxiv.org/abs/1809.08895)
            self.pos_embedding = ScaledPositionalEncoding(
                d_model=dim, dropout_rate=dropout, max_len=pe_max_len)
        else:
            self.pos_embedding = None

        self.use_class_token = use_class_token

        if self.use_class_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = Transformer(dim, depth, heads, dim_head, ff_dim, dropout, activation=activation)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
        )

        if mlp_head_hidden_dim:
            mlp_head_dim = list(mlp_head_hidden_dim)
            mlp_head_dim.insert(0, dim)
            for k in range(1, len(mlp_head_dim)):
                self.mlp_head.append(nn.Linear(mlp_head_dim[k-1], mlp_head_dim[k]))
                self.mlp_head.append(get_activation_fn(activation))
                self.mlp_head.append(nn.Dropout(mlp_head_dropout))
            self.mlp_head.append(nn.Linear(mlp_head_dim[-1], num_classes))
        else:
            # no hidden layers
            self.mlp_head.append(nn.Linear(dim, num_classes))

        # if num_classes==1:
        #     self.mlp_head.append(nn.Sigmoid())
            # self.mlp_head.append(nn.Flatten(start_dim=0))
        self.mlp_head.append(nn.Sigmoid())

    def forward(self, x):

        b, n, _ = x.shape

        if self.use_class_token:
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embedding is not None:
            x = self.pos_embedding(x)

        x = self.transformer(x)

        if self.use_class_token:
            x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        else:
            x = x.mean(dim=1)

        embedding = self.to_latent(x)
        # print(embedding.shape)
        score = self.mlp_head(embedding)

        return score

    ### get penultimate_embedding
    def emb(self, x):
        b, n, _ = x.shape

        if self.use_class_token:
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embedding is not None:
            x = self.pos_embedding(x)

        x = self.transformer(x)

        if self.use_class_token:
            x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        else:
            x = x.mean(dim=1)

        embedding = self.to_latent(x)


        return embedding      
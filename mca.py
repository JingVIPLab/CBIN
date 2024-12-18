# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from net_utils import FC, MLP, LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch, math


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, out_size, head):
        super(MHAtt, self).__init__()
        self.out = out_size

        self.linear_v = nn.Linear(512, out_size)
        self.linear_k = nn.Linear(512, out_size)
        self.linear_q = nn.Linear(512, out_size)
        self.linear_merge = nn.Linear(out_size, out_size)

        self.dropout = nn.Dropout()
        self.head = head

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(n_batches, -1, self.head, 128).transpose(1, 2)

        k = self.linear_k(k).view(n_batches, -1, self.head, 128).transpose(1, 2)

        q = self.linear_q(q).view(n_batches,-1, self.head, 128).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(n_batches, -1, self.out)

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, in_size, mid_size, out_size):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=in_size,
            mid_size=mid_size,
            out_size=out_size,
            dropout_r=0.1,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self):
        super(SA, self).__init__()

        self.mhatt = MHAtt(512, 4)
        self.ffn = FFN(512, 1024, 512)

        self.dropout1 = nn.Dropout(0.1)
        self.norm1 = LayerNorm(512)

        self.dropout2 = nn.Dropout(0.1)
        self.norm2 = LayerNorm(512)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(1024, 8)
        self.mhatt2 = MHAtt(1024, 8)
        self.ffn = FFN(1024, 1024, 1024)

        self.dropout1 = nn.Dropout(0.1)
        self.norm1 = LayerNorm(1024)

        self.dropout2 = nn.Dropout(0.1)
        self.norm2 = LayerNorm(1024)

        self.dropout3 = nn.Dropout(0.1)
        self.norm3 = LayerNorm(1024)
        self.linear = nn.Linear(512, 1024)


    # MHATT(V,K,Q)
    def forward(self, x, y, x_mask, y_mask):

        x = self.norm2(self.linear(x) + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x

# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, ):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA() for _ in range(1)])
        self.dec_list = nn.ModuleList([SGA() for _ in range(1)])


    # y:图像特征 ，x:语言特征
    def forward(self, x, y, x_mask, y_mask):
        # Get encoder last hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)

        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors
        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)

        return x, y

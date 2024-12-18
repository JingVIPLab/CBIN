
import torch.nn as nn
import torch.nn.functional as F
import torch
from mca import SA
from at import AT

class Question_Encoder(nn.Module):
    def __init__(self, pretrained_emb, token_size, emb_size=300, out_dim=300):
        super(Question_Encoder, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=emb_size,
        )

        # Loading the GloVe embedding weights
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
        self.embedding.requires_grad_(False)

        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=512,
            num_layers=1,
            batch_first=True
        )

        self.layer_last1 = nn.Sequential(nn.Linear(in_features=15*512,
                                                  out_features=out_dim,
                                                  bias=True),
                                        nn.BatchNorm1d(out_dim))

        self.ret = AT()

    def forward(self, all_que):

        embedding = self.embedding(all_que)
        lang_feat, _ = self.lstm(embedding)

        lang_feat = self.ret(lang_feat)

        lang_feat = lang_feat.reshape(lang_feat.size(0), -1)

        lang_feat = self.layer_last1(lang_feat)

        return lang_feat, embedding

    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)


class Cls_Encoder(nn.Module):
    def __init__(self, pretrained_emb, token_size, emb_size=300):
        super(Cls_Encoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=512,
            num_layers=1,
            batch_first=True
        )
        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=emb_size
        )

        # Loading the GloVe embedding weights
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.layer_last = nn.Sequential(nn.Linear(in_features=512,
                                                  out_features=300,
                                                  bias=True),
                                        nn.BatchNorm1d(300))

        self.att = SA()

    def forward(self, all_cls):

        embedding = self.embedding(all_cls)
        cls_feat, _ = self.lstm(embedding)

        for _ in range(2):
            cls_feat = self.att(cls_feat, None)

        cls_feat = cls_feat.reshape(-1, 512)

        cls_feat = self.layer_last(cls_feat)

        cls_feat = cls_feat.reshape(-1, 50, 300)

        return cls_feat, embedding
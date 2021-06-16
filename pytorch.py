import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy


class DeepSilencer(nn.Module):
    def __init__(self, L=200, K=5):
        '''
        A Pytorch Lightning refactoring of github.com/xy-chen16/DeepSilencer.

        Requires: sequence length, kmer length.
        Ensures:  end-to-end nn.Module for use with suitable loss and training loop.
        
        Simon Levine's modifications:
            - nn.embedding for learnable embedding (?)
            - ensemble with DNABert (?)
        
        Necl

        '''
        super(DeepSilencer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=64, embedding_dim=4)
        
        # Expect the following input shapes:
        # input_seq  = Input(shape=(4, L, 1))
        # input_kmer = Input(shape=(4**K,))

        self.kmer_net

        self.seq_net=nn.Sequential(nn.Conv2d())
        
        # =nn.Sequential(
        #     nn.Conv1d(in_channels=4, out_channels=16, kernel_size=16), # -> [64, 16, 1485]
        #     nn.ReLU(),
        #     nn.AdaptiveMaxPool1d(16), #-> [64, 16, 16]
        #     nn.Flatten(), #-> [64,256]
        #     nn.Linear(256,1), # -> [64,1] , a single logit output per instance.
        #     # nn.Sigmoid() -> no need for final activation since we will use nn.BCEWithLogitsLoss()
        # )

    def forward(self, x):
        y_seq = self.seq_net(self.embedding(x).view(64,4,1500))
        y_kmer = self.kmer_net(self.)

        return y
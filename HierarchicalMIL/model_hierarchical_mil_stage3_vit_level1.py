"""
Breast cancer stage prediction from pathological whole slide images with hierarchical image pyramid transformers.
Project developed under the "High Risk Breast Cancer Prediction Contest Phase 2" 
by Nightingale, Association for Health Learning & Inference (AHLI)
and Providence St. Joseph Health

Parts of code were took over and adapted from HIPT library.

https://github.com/mahmoodlab/HIPT/blob/master/2-Weakly-Supervised-Subtyping/models/model_utils.py

https://github.com/mahmoodlab/HIPT/blob/master/2-Weakly-Supervised-Subtyping/models/model_hierarchical_mil.py

Copyright (C) 2023 Zsolt Bedohazi, Andras Biricz, Istvan Csabai
"""


import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attn_Net_Gated(nn.Module):  ### original D: 256
    def __init__(self, L = 192, D = 256, dropout = False, n_classes = 1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            ##########################################x original was 0.25
            self.attention_a.append(nn.Dropout(0.8))
            self.attention_b.append(nn.Dropout(0.8))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x
    
    
######################################
# 3-Stage HIPT Implementation (With Local-Global Pretraining) #
######################################
class HIPT_LGP_FC_STAGE3ONLY(nn.Module):
    ################################################ original was 0.25
    def __init__(self, dropout=0.25, n_classes=5):
        super(HIPT_LGP_FC_STAGE3ONLY, self).__init__()
        #self.fusion = fusion
        size = 192
        
        ### Global Aggregation   ###x original nhead = 3
        self.global_phi = nn.Sequential(nn.Linear(192, 192), nn.ReLU(), nn.Dropout(2*dropout))
        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=192, nhead=3, dim_feedforward=64, dropout=dropout, activation='relu'
            ), 
            num_layers=2
        )
        self.global_attn_pool = Attn_Net_Gated(L=size, D=size, dropout=dropout, n_classes=1)
        self.global_rho = nn.Sequential(*[nn.Linear(size, size), nn.ReLU(), nn.Dropout(2*dropout)])

        self.classifier = nn.Linear(size, n_classes)
        
        

    def forward(self, h_4096, **kwargs):
        h_4096 = h_4096.squeeze(0)
        ### Global
        #print('Bag size:', h_4096.shape)
        h_4096 = self.global_phi(h_4096)
        h_4096 = self.global_transformer(h_4096.unsqueeze(1)).squeeze(1)
        A_4096, h_4096 = self.global_attn_pool(h_4096)  
        A_4096 = torch.transpose(A_4096, 1, 0)
        A_4096 = F.softmax(A_4096, dim=1) 
        h_path = torch.mm(A_4096, h_4096)
        h_WSI = self.global_rho(h_path)

        logits = self.classifier(h_WSI)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        return logits, F.softmax(logits, dim=1), Y_hat, None, None

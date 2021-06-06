"""
handle inputs as two features of slowfast and TSN respectively
"""
import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn.modules.container import Sequential
import torchvision.models as models

from torch.nn.modules import conv
from resnet import ResNet50, ResNet34, ResNet18
from config import *

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=FEATURE_LEN):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class CustomEncoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.pe = PositionalEncoding(d_model=out_channel)
        self.feature_reduction = nn.Conv1d(in_channel, out_channel, 1)
        self.long_layer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=out_channel, nhead=8), 4)

    def forward(self, x):
        '''
        IN: torch.FloatTensor [B,L,E]
        OUT: torch.FloatTensor [B,C,L,E']
        '''
        x = x.permute(0,2,1)
        x = self.feature_reduction(x)
        x = x.permute(2,0,1)
        x = self.pe(x)
        #[L,B,E]
        x = self.long_layer(x).permute(1,2,0)

        return x


class SJNET(nn.Module):
    def __init__(self, encoder_hidden=ENCODER_HIDDEN, decoder_hidden=DECODER_HIDDEN):
        super().__init__()
        self.encoder = CustomEncoder(FEATURE_DIM_2, encoder_hidden)
       
        # self.decoder = nn.RNN(input_size=FEATURE_DIM, 
        self.decoder = nn.GRU(input_size=encoder_hidden, 
                              hidden_size=encoder_hidden // 2, 
                              num_layers=2, 
                              batch_first=True,
                              dropout=DROP_RATE,
                              bidirectional=True)

        self.out = nn.Sequential(
            nn.Conv1d(encoder_hidden, decoder_hidden, 1),
            nn.ReLU(),
            nn.Dropout(DROP_RATE),
            # nn.Conv1d(decoder_hidden, decoder_hidden, 1),
            # nn.ReLU(),
            nn.Conv1d(decoder_hidden, 1, 1))
        
        # HERE - change optimizer
        self.opt = torch.optim.AdamW(self.parameters(), lr=LEARNING_RATE)
        self.to(DEVICE)
        
    def forward(self, x):
        #IN: [B, L, E]
        x = self.encoder(x)

        x = x.permute(0, 2, 1)
        x, _ = self.decoder(x)
        x = x.permute(0, 2, 1)

        x = self.out(x).squeeze()
        # x_sf = self.out_sf(x_sf).squeeze()
        # x_tsn = self.out_tsn(x_tsn).squeeze()
        
        # x = x_sf + x_tsn

        return x


class JWNET(nn.Module):
    def __init__(self, in_channel=FEATURE_DIM_1, encoder_hidden=ENCODER_HIDDEN, decoder_hidden=DECODER_HIDDEN):
        super().__init__()

        self.out = nn.Sequential(
            nn.Conv1d(in_channel, encoder_hidden, 7, padding=3),
            nn.ReLU(),
            nn.Dropout(DROP_RATE),
            nn.Conv1d(encoder_hidden, decoder_hidden, 7, padding=3),
            nn.ReLU(),
            nn.Dropout(DROP_RATE),
            nn.Conv1d(decoder_hidden, decoder_hidden, 1),
            nn.ReLU(),
            nn.Dropout(DROP_RATE),
            nn.Conv1d(decoder_hidden, 1, 1))

        # HERE - change optimizer
        self.opt = torch.optim.AdamW(self.parameters(), lr=LEARNING_RATE)
        self.to(DEVICE)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.out(x).squeeze()
        return x





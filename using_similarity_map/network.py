import torch
import torch.nn as nn
import torch.distributions as distributions
import numpy as np
import math

from resnet import ResNet18
from config import *


def pairwise_cosine_similarity(x, y):
    '''
    calculate self-pairwise cosine distance
    input:
    x: torch.FloatTensor [B,C,L,E']
    y: torch.FloatTensor [B,C,L,E']
    output:
    xcos_dist: torch.FloatTensor [B,C,L,L]
    '''
    x = x.detach()
    y = y.permute(0,1,3,2)
    dot = torch.matmul(x, y)
    x_dist = torch.norm(x, p=2, dim=3, keepdim=True)
    y_dist = torch.norm(y, p=2, dim=2, keepdim=True)
    dist = x_dist * y_dist
    cos = dot / (dist + 1e-8)
    #cos_dist = 1 - cos
    return cos

def pairwise_minus_l2_distance(x, y):
    '''
    sim-siam style
    calculate pairwise l2 distance
    input:
    x: torch.FloatTensor [B,C,L,E']
    y: torch.FloatTensor [B,C,L,E']
    output:
    -l2_dist: torch.FloatTensor [B,C,L,L]
    '''
    '''
    x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    y_norm = torch.norm(y, p=2, dim=-1, keepdim=True)
    x = x/x_norm
    y = y/y_norm
    '''
    x = x.unsqueeze(3).detach()
    y = y.unsqueeze(2)
    l2_dist = torch.sqrt(torch.sum((x-y)**2, dim=-1) + 1e-8)
    l2_dist = nn.InstanceNorm2d(l2_dist.size(1))(l2_dist)
    return -l2_dist

def get_tsm_x(out, out_projection_heads):
    out_projection_list = []
    for i in range(out.size(1)):
        out_projection_list.append(out_projection_heads[i](out[:,i,:,:].permute(0,2,1)))
    projected_out = torch.stack(out_projection_list, dim=1).permute(0,1,3,2)
    tsm = pairwise_minus_l2_distance(out, projected_out)
    x = pairwise_minus_l2_distance(out, out)
    return tsm, x

def get_direct_logit(out_list, out_projection_head):
    #detached!
    tmp_list = []
    for out in out_list:
        out = out.permute(0,1,3,2).contiguous()
        out = out.view(out.size(0), -1, out.size(-1))
        tmp_list.append(out)
    out = torch.cat(tmp_list, dim=1)
    projected_out = out_projection_head(out).squeeze()
    return projected_out

def get_direct_logit_transformer(out_list, feature_reduction, positional_encoding, transformer, final_head):
    #detached!
    tmp_list = []
    for out in out_list:
        out = out.permute(0,1,3,2).contiguous()
        out = out.view(out.size(0), -1, out.size(-1))
        tmp_list.append(out)
    out = torch.cat(tmp_list, dim=1).detach()
    # out = torch.cat(tmp_list, dim=1)
    reduced_feature = positional_encoding(torch.relu(feature_reduction(out).permute(2,0,1)))
    out = final_head(transformer(reduced_feature).permute(1,2,0)).squeeze()
    return out
    #B,C,L

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
    def __init__(self, hidden, channel_num):
        super().__init__()
        self.feature_reduction = nn.Sequential(
            nn.Conv1d(FEATURE_DIM, hidden, 1),
            nn.PReLU()
        )
        self.pe = PositionalEncoding(d_model=hidden)
        self.event_short_layer = nn.Sequential(
            nn.Conv1d(hidden, hidden, 1),
            nn.PReLU(),
            nn.Conv1d(hidden, hidden, 1),
        )
        self.shot_short_layer = nn.Sequential(
            nn.Conv1d(hidden, hidden, 1),
            nn.PReLU(),
            nn.Conv1d(hidden, hidden, 1),
        )
        self.whole_short_layer = nn.Sequential(
            nn.Conv1d(hidden, hidden, 1),
            nn.PReLU(),
            nn.Conv1d(hidden, hidden, 1),
        )
        self.event_middle_layer_1 = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.PReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
        )
        self.shot_middle_layer_1 = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.PReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
        )
        self.whole_middle_layer_1 = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.PReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
        )
        self.event_middle_layer_2 = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.PReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
        )
        self.shot_middle_layer_2 = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.PReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
        )
        self.whole_middle_layer_2 = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.PReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
        )
        #transformer
        self.event_long_layer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden, nhead=4), 3)
        self.shot_long_layer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden, nhead=4), 3)
        self.whole_long_layer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden, nhead=4), 3)
        
        

    def forward(self, x):
        '''
        IN: torch.FloatTensor [B,L,E]
        OUT: torch.FloatTensor [B,C,L,E']
        '''
        
        #first, do conv
        x = x.permute(0,2,1)
        #[B,E,L]
        reduced_feature = self.feature_reduction(x)
        #[B,E,L], event
        event_short = 0.5*(self.event_short_layer(reduced_feature) + reduced_feature)
        event_middle_1 = 0.5*(self.event_middle_layer_1(reduced_feature) + reduced_feature)
        event_middle_2 = 0.5*(self.event_middle_layer_2(event_middle_1) + event_middle_1)
        #do transformer computation
        event_long = reduced_feature.permute(2,0,1)
        event_long = self.pe(event_long)
        #[L,B,E]
        event_long = self.event_long_layer(event_long).permute(1,2,0)
        event_out_list = [event_short, event_middle_1, event_middle_2, event_long]
        event_out = torch.stack(event_out_list, dim=1).permute(0,1,3,2)
        #[B,E,L], shot
        shot_short = 0.5*(self.shot_short_layer(reduced_feature) + reduced_feature)
        shot_middle_1 = 0.5*(self.shot_middle_layer_1(reduced_feature) + reduced_feature)
        shot_middle_2 = 0.5*(self.shot_middle_layer_2(shot_middle_1) + shot_middle_1)
        #do transformer computation
        shot_long = reduced_feature.permute(2,0,1)
        shot_long = self.pe(shot_long)
        #[L,B,E]
        shot_long = self.shot_long_layer(shot_long).permute(1,2,0)
        shot_out_list = [shot_short, shot_middle_1, shot_middle_2, shot_long]
        shot_out = torch.stack(shot_out_list, dim=1).permute(0,1,3,2)
        #[B,E,L], whole
        whole_short = 0.5*(self.whole_short_layer(reduced_feature) + reduced_feature)
        whole_middle_1 = 0.5*(self.whole_middle_layer_1(reduced_feature) + reduced_feature)
        whole_middle_2 = 0.5*(self.whole_middle_layer_2(whole_middle_1) + whole_middle_1)
        #do transformer computation
        whole_long = reduced_feature.permute(2,0,1)
        whole_long = self.pe(whole_long)
        #[L,B,E]
        whole_long = self.whole_long_layer(whole_long).permute(1,2,0)
        whole_out_list = [whole_short, whole_middle_1, whole_middle_2, whole_long]
        whole_out = torch.stack(whole_out_list, dim=1).permute(0,1,3,2)
        return event_out, shot_out, whole_out


# Temporary name for the network
class SJNET(nn.Module):
    def __init__(self, encoder_hidden=ENCODER_HIDDEN, channel_num=CHANNEL_NUM, decoder_hidden=DECODER_HIDDEN):
        super().__init__()
        self.encoder = CustomEncoder(encoder_hidden, channel_num)
        self.event_projection_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Conv1d(encoder_hidden, encoder_hidden, 1),
                    nn.PReLU(),
                    nn.Conv1d(encoder_hidden, encoder_hidden, 1),
                ) 
                for _ in range(CHANNEL_NUM)
            ]
        )
        self.shot_projection_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Conv1d(encoder_hidden, encoder_hidden, 1),
                    nn.PReLU(),
                    nn.Conv1d(encoder_hidden, encoder_hidden, 1),
                ) 
                for _ in range(CHANNEL_NUM)
            ]
        )
        self.whole_projection_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Conv1d(encoder_hidden, encoder_hidden, 1),
                    nn.PReLU(),
                    nn.Conv1d(encoder_hidden, encoder_hidden, 1),
                ) 
                for _ in range(CHANNEL_NUM)
            ]
        )
        self.class_projection_head = nn.Sequential(
            nn.Conv1d(encoder_hidden*CHANNEL_NUM*3, encoder_hidden*CHANNEL_NUM, 5, padding=2),
            nn.PReLU(),
            nn.Conv1d(encoder_hidden*CHANNEL_NUM, encoder_hidden, 5, padding=2),
            nn.PReLU(),
            nn.Conv1d(encoder_hidden, encoder_hidden//2, 5, padding=2),
            nn.PReLU(),
            nn.Conv1d(encoder_hidden//2, 1, 1),
        )
        self.class_feature_reduction = nn.Conv1d(encoder_hidden*CHANNEL_NUM*3, encoder_hidden, 1)
        self.class_positional_encoding = PositionalEncoding(d_model=encoder_hidden)
        self.class_transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=encoder_hidden, nhead=4), 3)
        self.class_final_head = nn.Conv1d(encoder_hidden, 1, 1)

        self.conv2d = ResNet18(channel_num*3, decoder_hidden)
        self.tsm_decoder = nn.Sequential(
            nn.Conv1d(decoder_hidden, decoder_hidden, 5, padding=2),
            nn.PReLU(),
            nn.Conv1d(decoder_hidden, decoder_hidden, kernel_size=5, padding=2),
            nn.PReLU(),
        )
        self.tsm_out = nn.Sequential(
            nn.Conv1d(decoder_hidden, decoder_hidden, 1),
            nn.PReLU(),
            nn.Conv1d(decoder_hidden, decoder_hidden, 1),
            nn.PReLU(),
            nn.Conv1d(decoder_hidden, 1, 1),
        )
        self.alpha = nn.parameter.Parameter(torch.zeros(1,))
        self.opt = torch.optim.AdamW(self.parameters(), lr=LEARNING_RATE)
        self.to(DEVICE)
        
    def forward(self, x):
        #IN: [B, L, E]
        event_out, shot_out, whole_out = self.encoder(x)
        event_tsm, event_x = get_tsm_x(event_out, self.event_projection_heads)
        shot_tsm, shot_x = get_tsm_x(shot_out, self.shot_projection_heads)
        whole_tsm, whole_x = get_tsm_x(whole_out, self.whole_projection_heads)
        direct_score = get_direct_logit_transformer([event_out, shot_out, whole_out], self.class_feature_reduction, self.class_positional_encoding, self.class_transformer, self.class_final_head)
        x = torch.cat([event_x, shot_x, whole_x], dim=1)
        x = self.conv2d(x)
        tsm_score = torch.diagonal(x, dim1=2, dim2=3)
        tsm_score = self.tsm_decoder(tsm_score)
        tsm_score = self.tsm_out(tsm_score).squeeze()
        return tsm_score, direct_score, event_tsm, shot_tsm, whole_tsm

    def get_tsm(self, x):
        with torch.no_grad():
            event_out, shot_out, whole_out = self.encoder(x)
            event_x = pairwise_minus_l2_distance(event_out, event_out)
            shot_x = pairwise_minus_l2_distance(shot_out, shot_out)
            whole_x = pairwise_minus_l2_distance(whole_out, whole_out)

            x = torch.cat([event_x, shot_x, whole_x], dim=1)
            x = torch.mean(x, dim=1).unsqueeze(1)
            #for print, we permute x
            x = x.permute(0,2,3,1)
            x = x.detach().cpu().numpy()
        return x

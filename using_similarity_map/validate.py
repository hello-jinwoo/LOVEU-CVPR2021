import pickle
import os
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from network import SJNET
from validation import validate
from dataset import Kinetics_GEBD_train, Kinetics_GEBD_validation, Kinetics_GEBD_test
from tqdm import tqdm
from config import *
from torch.multiprocessing import Pool, Process, set_start_method, cpu_count
import warnings
import argparse

warnings.filterwarnings("ignore")
device = torch.device('cuda')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='')
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()

    validation_dataloader = DataLoader(Kinetics_GEBD_validation(args.fold), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    network = torch.load(args.model)

    network.eval()

    f1_results = {}
    prec_results = {}
    rec_results = {}
    val_dicts = {}

    s = args.sigma
    val_dict = {}
    k = 3
    gaussian_filter = torch.FloatTensor(
                                [np.exp(-z*z/(2*s*s))/np.sqrt(2*np.pi*s*s) for z in range(-k, k+1)]
                                ).to(device)
    gaussian_filter = gaussian_filter.unsqueeze(0).unsqueeze(0)
    gaussian_filter /= torch.max(gaussian_filter)
    gaussian_filter = gaussian_filter.repeat(1, FEATURE_LEN, 1)
    max_pooling = nn.MaxPool1d(5, stride=1, padding=2)

    for feature, filenames, durations in tqdm(validation_dataloader):
        feature = feature.to(device)
        with torch.no_grad():
            pred1, pred2, _, _, _ = network(feature)
            alpha = torch.sigmoid(network.module.alpha).unsqueeze(0)
            pred = alpha*pred1.detach() + (1-alpha)*pred2.detach()
            pred = torch.sigmoid(pred) # [BATCH_SIZE, FEATURE_LEN]

            if s > 0:
                out = pred.unsqueeze(-1)
                eye = torch.eye(FEATURE_LEN).to(device)
                out = out * eye
                out = nn.functional.conv1d(out, gaussian_filter, padding=k)
            else:
                out = pred.unsqueeze(1)

            peak = (out == max_pooling(out))
            peak[out < THRESHOLD] = False
            peak = peak.squeeze()

            idx = torch.nonzero(peak).cpu().numpy()
            
        durations = durations.numpy()

        boundary_list = [[] for _ in range(len(out))]
        for i, j in idx:
            duration = durations[i]
            first = TIME_UNIT/2
            if first + TIME_UNIT*j < duration:
                boundary_list[i].append(first + TIME_UNIT*j)
        for i, boundary in enumerate(boundary_list):
            filename = filenames[i]
            val_dict[filename] = boundary

    val_dicts[s] = val_dict
    f1, prec, rec = validate(val_dict, args.fold)
    f1_results[s] = f1
    prec_results[s] = prec
    rec_results[s] = rec

    print(f'f1: {f1_results}')
    print(f'precision: {prec_results}')
    print(f'recall: {rec_results}')
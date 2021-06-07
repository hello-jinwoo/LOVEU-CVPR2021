import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from dataset import Kinetics_GEBD_test
from torch.utils.data import DataLoader
from config import *
import pickle 
import argparse

device = torch.device('cuda')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ver', default='')
    args = parser.parse_args()

    test_dataloader = DataLoader(Kinetics_GEBD_test(), batch_size=BATCH_SIZE, shuffle=False)

    print("PROB RESULTS LIST:", os.listdir(PROB_RESULT_PATH))
    
    prob_result_pkl_list = []
    for i, prob_result_file in enumerate(os.listdir(PROB_RESULT_PATH)):
        print(i)
        with open(os.path.join(PROB_RESULT_PATH, prob_result_file), 'rb') as f: 
            prob_result_pkl_list.append(pickle.load(f))
    len_results = len(prob_result_pkl_list)
    print(len_results)

    max_pooling = nn.MaxPool1d(POOL_SIZE, stride=1, padding=POOL_SIZE // 2)

    test_dict = {}
    for feature, filenames, durations in tqdm(test_dataloader):
        out = torch.zeros([len(feature), FEATURE_LEN]).to(device) # [B, L]
        with torch.no_grad():
            for prob_result_pkl in prob_result_pkl_list:
                out += torch.cat([prob_result_pkl[filename] for filename in filenames]) # [B, L]
            out /= len_results

            out = out.unsqueeze(1)

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
            test_dict[filename] = boundary
    with open('results/test_ensemble_' + args.ver + '.pkl', 'wb') as f:
        pickle.dump(test_dict, f)
    
    print("TEST ENDS!")

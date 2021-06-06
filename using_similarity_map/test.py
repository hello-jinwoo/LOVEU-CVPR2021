import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from dataset import Kinetics_GEBD_validation, Kinetics_GEBD_test
from torch.utils.data import DataLoader
from config import *
from validation import validate

device = torch.device('cuda')

if __name__ == "__main__":
    validation_dataloader = DataLoader(Kinetics_GEBD_validation(), batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(Kinetics_GEBD_test(), batch_size=BATCH_SIZE, shuffle=False)

    print("MODEL LIST:", os.listdir(MODEL_SAVE_PATH))

    network_list = []
    for model in os.listdir(MODEL_SAVE_PATH):
        load_model = torch.load(os.path.join(MODEL_SAVE_PATH, model)).to(device)
        load_model.eval()
        network_list.append(load_model)

    f1_results = {}
    prec_results = {}
    rec_results = {}
    val_dicts = {}
    k = 3
    max_key = k
    max_value = 0
    print("sigma list:", SIGMA_LIST)
    for s in SIGMA_LIST:
        val_dict = {}
        gaussian_filter = torch.FloatTensor(
            [np.exp(-z*z/(2*s*s))/np.sqrt(2*np.pi*s*s) for z in range(-k, k+1)]
        ).to(device)
        gaussian_filter = gaussian_filter.unsqueeze(0).unsqueeze(0)
        gaussian_filter /= torch.max(gaussian_filter)
        gaussian_filter = gaussian_filter.repeat(1, FEATURE_LEN, 1)
        max_pooling = nn.MaxPool1d(5, stride=1, padding=2)
        print("validation for s", s)
        for feature, filenames, durations in tqdm(validation_dataloader):
            feature = feature.to(device)
            with torch.no_grad():
                pred = torch.zeros([feature.shape[0], FEATURE_LEN]).to(device)
                for network in network_list:
                    pred_tmp, _ = network(feature)
                    pred_tmp = torch.sigmoid(pred_tmp) # [BATCH_SIZE, FEATURE_LEN]
                    pred += pred_tmp
                pred /= len(network_list)

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
        f1, prec, rec = validate(val_dict)
        f1_results[s] = f1
        prec_results[s] = prec
        rec_results[s] = rec

        if f1 > max_value:
            max_value = f1
            max_key = s

    print("VALIDATION RESULTS!")
    print(f'f1: {f1_results}')
    print(f'precision: {prec_results}')
    print(f'recall: {rec_results}')
    print()

    #test
    s = max_key
    val_dict = val_dicts[max_key]
    with open('val.pkl', 'wb') as f:
        pickle.dump(val_dict, f)
    test_dict = {}
    
    print("TEST STARTS!")
    for feature, filenames, durations in tqdm(test_dataloader):
        feature = feature.to(device)
        with torch.no_grad():
            pred = torch.zeros([feature.shape[0], FEATURE_LEN]).to(device)
            for network in network_list:
                pred_tmp, _ = network(feature)
                pred_tmp = torch.sigmoid(pred_tmp) # [BATCH_SIZE, FEATURE_LEN]
                pred += pred_tmp
            pred /= len(network_list)

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
            test_dict[filename] = boundary
    with open('results/test_ensemble_' + str(max_value)[2:6] '.pkl', 'wb') as f:
        pickle.dump(test_dict, f)
    
    print("TEST ENDS!")

import pickle
import os
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from network import SJNET, JWNET
from validation import validate
from dataset import Kinetics_GEBD_train, Kinetics_GEBD_validation, Kinetics_GEBD_test
from tqdm import tqdm
from config import *
from torch.multiprocessing import Pool, Process, set_start_method, cpu_count
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda')

def get_max_key_value(dictionary: dict):
    keys = dictionary.keys()
    max_value = -1
    max_key = -1
    for key in keys:
        if dictionary[key] > max_value:
            max_key = key
            max_value = dictionary[key]
    return max_key, max_value

if __name__ == '__main__':
    try:
        set_start_method('spawn')
    except RuntimeError as e:
        print(e)
        pass

    torch.set_printoptions(sci_mode=False)

    test_dataloader = DataLoader(Kinetics_GEBD_test(), batch_size=BATCH_SIZE, shuffle=False, num_workers=2) 

    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCELoss()
    sigmoid = nn.Sigmoid().to(device)
    
    for fold in range(5):
        fold_done_flag = False
        test_threshold = TEST_THRESHOLD
        print(f"< FOLD {fold} >")
        # for early stopping
        no_improvement_duration = 0
        val_max_f1 = 0
        improve_flag = True

        train_dataloader = DataLoader(Kinetics_GEBD_train(fold), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        validation_dataloader = DataLoader(Kinetics_GEBD_validation(fold), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        network_sf = nn.DataParallel(JWNET()).to(device) # [JW-add] DataParallel for multi gpu implementation
        network_tsn = nn.DataParallel(SJNET()).to(device) # [JW-add] DataParallel for multi gpu implementation

        for epoch in range(250):
            # goal accomplishment check
            if fold_done_flag:
                print(f"GOAL({GOAL_SCORE})_ACCOMPLISHED!")
                print()
                break

            # check early stopping
            if not improve_flag:
                no_improvement_duration += 1
                if no_improvement_duration >= PATIENCE:
                    print("EARLY STOPPING !!")
                    print()
                    break
            improve_flag = False

            epoch_loss_list = []
            network_sf.train()
            network_tsn.train()
            
            for feature, annotations, annotations_soft in tqdm(train_dataloader):
                feature = feature.to(device)
                annotations = annotations.to(device)

                n = random.randint(0,9999)
                
                out_sf = network_sf(feature[..., :FEATURE_DIM_1])
                out_tsn = network_tsn(feature[..., FEATURE_DIM_1:])

                answer_idx = torch.randint(0, 5, (len(out_sf),))
                annotations = annotations[range(len(answer_idx)), answer_idx, :]
                
                loss_sf = criterion(out_sf, annotations)
                loss_tsn = criterion(out_tsn, annotations)

                network_sf.module.opt.zero_grad() # [JW-add] add .module to deal with DataParallel
                network_tsn.module.opt.zero_grad() # [JW-add] add .module to deal with DataParallel
                
                loss_sf.backward()
                loss_tsn.backward()
                
                torch.nn.utils.clip_grad_norm_(network_sf.parameters(), max_norm=2.0)
                torch.nn.utils.clip_grad_norm_(network_tsn.parameters(), max_norm=2.0)
                
                network_sf.module.opt.step() # [JW-add] add .module to deal with DataParallel
                network_tsn.module.opt.step() # [JW-add] add .module to deal with DataParallel
                

            network_sf.eval()
            network_tsn.eval()

            f1_results = {}
            prec_results = {}
            rec_results = {}
            val_dicts = {}
            for s in SIGMA_LIST:
                val_dict = {}
                k = 3
                gaussian_filter = torch.FloatTensor(
                                            [np.exp(-z*z/(2*s*s))/np.sqrt(2*np.pi*s*s) for z in range(-k, k+1)]
                                            ).to(device)
                gaussian_filter = gaussian_filter.unsqueeze(0).unsqueeze(0)
                gaussian_filter /= torch.max(gaussian_filter)
                gaussian_filter = gaussian_filter.repeat(1, FEATURE_LEN, 1)
                max_pooling = nn.MaxPool1d(5, stride=1, padding=2)

                for feature, filenames, durations in validation_dataloader:
                    feature = feature.to(device)
                    with torch.no_grad():
                        pred_sf = network_sf(feature[..., :FEATURE_DIM_1])
                        pred_tsn = network_tsn(feature[..., FEATURE_DIM_1:])
                        pred_sf = torch.sigmoid(pred_sf) # [BATCH_SIZE, FEATURE_LEN]
                        pred_tsn = torch.sigmoid(pred_tsn) # [BATCH_SIZE, FEATURE_LEN]
                        pred = (pred_sf + pred_tsn) / 2

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
                f1, prec, rec = validate(val_dict, fold)
                f1_results[s] = f1
                prec_results[s] = prec
                rec_results[s] = rec
                if f1 > val_max_f1:
                    val_max_f1 = f1
                    improve_flag = True
                    no_improvement_duration = 0

            print(f'epoch: {epoch+1}, f1: {f1_results}')
            print(f'epoch: {epoch+1}, precision: {prec_results}')
            print(f'epoch: {epoch+1}, recall: {rec_results}')
            
            #test
            max_key, max_value = get_max_key_value(f1_results)
            s = max_key
            if max_value > test_threshold:
                if max_value > GOAL_SCORE:
                    fold_done_flag = True
                test_threshold = max_value + 0.0005

                description = f'kim_fold_{fold}_s_{s}_'
                if 'all' in DATA_PATH:
                    description += 'all_'
                elif 'both' in DATA_PATH:
                    description += 'both'

                print(f'conducting test! : val-f1: {f1_results[max_key]}')
                val_dict = val_dicts[max_key]
                with open(f'results/val_' + description + str(max_value)[2:6] + '.pkl', 'wb') as f: 
                    pickle.dump(val_dict, f)
                
                test_dict = {}
                test_prob_dict = {}
                k = 3
                gaussian_filter = torch.FloatTensor(
                    [np.exp(-z*z/(2*s*s))/np.sqrt(2*np.pi*s*s) for z in range(-k, k+1)]
                ).to(DEVICE)
                gaussian_filter = gaussian_filter.unsqueeze(0).unsqueeze(0)
                gaussian_filter /= torch.max(gaussian_filter)
                gaussian_filter = gaussian_filter.repeat(1, FEATURE_LEN, 1)
                max_pooling = nn.MaxPool1d(5, stride=1, padding=2)
                
                for feature, filenames, durations in test_dataloader:
                    feature = feature.to(DEVICE)
                    with torch.no_grad():
                        pred_sf = network_sf(feature[..., :FEATURE_DIM_1])
                        pred_tsn = network_tsn(feature[..., FEATURE_DIM_1:])
                        pred_sf = torch.sigmoid(pred_sf) # [BATCH_SIZE, FEATURE_LEN]
                        pred_tsn = torch.sigmoid(pred_tsn) # [BATCH_SIZE, FEATURE_LEN]
                        pred = (pred_sf + pred_tsn) / 2

                        if s > 0:
                            out = pred.unsqueeze(-1)
                            eye = torch.eye(FEATURE_LEN).to(DEVICE)
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
                        test_prob_dict[filename] = out[i]

                with open(f'results/test_' + description + str(max_value)[2:6] + '.pkl', 'wb') as f:
                    pickle.dump(test_dict, f)
                with open(f'prob_results/test_prob_' + description + str(max_value)[2:6] + '.pkl', 'wb') as f:
                    pickle.dump(test_prob_dict, f)
                torch.save(network_sf, f'{MODEL_SAVE_PATH}/model_sf_' + description + str(max_value)[2:6] + '.pt')
                torch.save(network_tsn, f'{MODEL_SAVE_PATH}/model_tsn_' + description + str(max_value)[2:6] + '.pt')

        



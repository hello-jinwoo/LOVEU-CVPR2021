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

import time

device = torch.device('cuda')

import warnings
warnings.filterwarnings("ignore")

def get_max_key_value(dictionary: dict):
    keys = dictionary.keys()
    max_value = -1
    max_key = -1
    for key in keys:
        if dictionary[key] > max_value:
            max_key = key
            max_value = dictionary[key]
    return max_key, max_value

def get_basic_mask(gap=5):
    tmp = [[i,j] for i in range(FEATURE_LEN) for j in range(FEATURE_LEN) if np.abs(i-j) <= gap]
    x, y = zip(*tmp)
    basic_mask = np.zeros((FEATURE_LEN, FEATURE_LEN))
    basic_mask[x, y]= 1
    basic_mask = torch.from_numpy(basic_mask.astype(np.bool))
    return basic_mask

def get_mask(tsm, annotations):
    indices = torch.nonzero(annotations).cpu().numpy()
    mask = torch.eye(tsm.size(2)).unsqueeze(0).repeat(tsm.size(0), 1, 1)
    current_batch = 0
    current_start = 0
    for i,j in indices:
        if i != current_batch:
            mask[current_batch, current_start:, current_start:] = 1
            current_batch = i
            current_start = 0
        mask[i, current_start:j, current_start:j] = 1
        current_start = j+1
    return mask.bool().unsqueeze(1)

if __name__ == '__main__':
    try:
        set_start_method('spawn')
    except RuntimeError as e:
        print(e)
        pass

    torch.set_printoptions(threshold=np.inf, sci_mode=False)
    torch.autograd.set_detect_anomaly(True)
    test_dataloader = DataLoader(Kinetics_GEBD_test(), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    _criterion = nn.BCEWithLogitsLoss(reduction='none')
    criterion = nn.BCEWithLogitsLoss()
    basic_mask = get_basic_mask(gap=GAP)
    loss_list = []

    for fold in range(5):
        network = nn.DataParallel(SJNET()).to(device)

        fold_done_flag = False
        test_threshold = TEST_THRESHOLD
        print(f"< FOLD {fold} >", fold_list)
        # for early stopping
        no_improvement_duration = 0
        val_max_f1 = 0
        improve_flag = True

        train_dataloader = DataLoader(Kinetics_GEBD_train(fold), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        validation_dataloader = DataLoader(Kinetics_GEBD_validation(fold), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


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
            network.train()

            for feature, event_boundaries, shot_boundaries, whole_boundaries in tqdm(train_dataloader):
                feature = feature.to(device)
                event_boundaries = event_boundaries.to(device)
                shot_boundaries = shot_boundaries.to(device)
                whole_boundaries = whole_boundaries.to(device)
                n = random.randint(0,9999)

                tsm_out, direct_out, event_tsm, shot_tsm, whole_tsm = network(feature)

                answer_idx = torch.randint(0, 5, (len(tsm_out),))
                
                event_boundaries = event_boundaries[range(len(answer_idx)), answer_idx, :]
                shot_boundaries = shot_boundaries[range(len(answer_idx)), answer_idx, :]
                whole_boundaries = whole_boundaries[range(len(answer_idx)), answer_idx, :]
                _basic_mask = basic_mask.repeat(event_tsm.size(0), event_tsm.size(1),1,1)

                #whole_tsm loss
                whole_mask = get_mask(whole_tsm, whole_boundaries)
                whole_mask = whole_mask.repeat(1,CHANNEL_NUM,1,1)
                whole_anti_mask = torch.logical_not(whole_mask)
                whole_mask = torch.logical_and(_basic_mask, whole_mask)
                whole_anti_mask = torch.logical_and(_basic_mask, whole_anti_mask)
                whole_aux_loss = WHOLE_LOSS_COEF*(torch.mean(whole_tsm[whole_anti_mask]) - torch.mean(whole_tsm[whole_mask]))

                #shot_tsm_loss
                shot_mask = get_mask(shot_tsm, shot_boundaries)
                shot_mask = shot_mask.repeat(1,CHANNEL_NUM,1,1)
                shot_anti_mask = torch.logical_not(shot_mask)
                shot_mask = torch.logical_and(_basic_mask, shot_mask)
                shot_anti_mask = torch.logical_and(_basic_mask, shot_anti_mask)
                shot_aux_loss = SHOT_LOSS_COEF*(torch.mean(shot_tsm[shot_anti_mask]) - torch.mean(shot_tsm[shot_mask]))

                #event_tsm loss
                event_mask = get_mask(event_tsm, event_boundaries)
                event_mask = event_mask.repeat(1,CHANNEL_NUM,1,1)
                event_anti_mask = torch.logical_not(event_mask)
                event_mask = torch.logical_and(
                    torch.logical_xor(
                        torch.logical_and(_basic_mask, event_mask), shot_anti_mask
                    ), 
                    torch.logical_and(_basic_mask, event_mask)
                )
                event_anti_mask = torch.logical_and(
                    torch.logical_xor(
                        torch.logical_and(_basic_mask, event_anti_mask), shot_anti_mask
                    ), 
                    torch.logical_and(_basic_mask, event_anti_mask)
                )
                event_aux_loss = EVENT_LOSS_COEF*(torch.mean(event_tsm[event_anti_mask]) - torch.mean(event_tsm[event_mask]))
                alpha = torch.sigmoid(network.module.alpha).unsqueeze(0)
                final_score = alpha*tsm_out.detach() + (1-alpha)*direct_out.detach()

                loss = AUX_LOSS_COEF*(event_aux_loss+shot_aux_loss+whole_aux_loss) \
                + criterion(tsm_out, whole_boundaries) \
                + criterion(direct_out, whole_boundaries) \
                + criterion(final_score, whole_boundaries)
                # network.opt.zero_grad()
                network.module.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=2.0)
                network.module.alpha.grad *= 1000
                # network.opt.step()
                network.module.opt.step()
                epoch_loss_list.append(loss.detach().cpu().numpy())
            
            network.eval()
            
            loss_list.append(sum(epoch_loss_list)/len(epoch_loss_list))
            fig = plt.figure(figsize=(8,8))
            fig.add_subplot(2,3,1)
            plt.plot(list(range(len(loss_list))), loss_list)
            
            vis_batch = []               
            for vid_name in VAL_VIDEOS:
                path = f'{VISUAL_DATA_PATH}{vid_name}'
                vis_batch.append(np.load(path))
            vis_batch = torch.from_numpy(np.stack(vis_batch, axis=0)).float().to(device)
            sims = (network.module.get_tsm(vis_batch) + 1)/2
            for i in range(2,7):
                fig.add_subplot(2,3,i)
                plt.imshow(sims[i-2])
            plt.savefig(f'fold_{fold}.jpg')
            plt.clf()
            
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

                description = f'kang_v2_fold_{fold}_s_{s}_'
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
                ).to(device)
                gaussian_filter = gaussian_filter.unsqueeze(0).unsqueeze(0)
                gaussian_filter /= torch.max(gaussian_filter)
                gaussian_filter = gaussian_filter.repeat(1, FEATURE_LEN, 1)
                max_pooling = nn.MaxPool1d(5, stride=1, padding=2)
                
                for feature, filenames, durations in test_dataloader:
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
                        test_dict[filename] = boundary
                        test_prob_dict[filename] = out[i]

                with open(f'results/test_' + description + str(max_value)[2:6] + '.pkl', 'wb') as f:
                    pickle.dump(test_dict, f)
                with open(f'prob_results/test_prob_' + description + str(max_value)[2:6] + '.pkl', 'wb') as f:
                    pickle.dump(test_prob_dict, f)
                torch.save(network, f'public_models/model_' + description + str(max_value)[2:6] + '.pt')
            



    


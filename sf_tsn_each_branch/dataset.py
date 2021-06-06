import os
import pickle
import json
import random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from config import *

class Kinetics_GEBD_train(Dataset):
    def __init__(self, n_fold):
        with open(ANNOTATION_PATH, 'rb') as f:
            self.annotations = pickle.load(f)
        with open(FILE_LIST, 'rb') as f:
            self.filenames = pickle.load(f)[n_fold]['train']

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        tmp = random.randint(0,10000)
        try:
            path1 = os.path.join(DATA_PATH_2, self.filenames[idx])
            f1 = torch.from_numpy(np.load(path1)).float()
        except:
            tmp = 10000
        if tmp < 10000 * INTERPOLATION_PROB:
            path1 = os.path.join(DATA_PATH_2, self.filenames[idx])
            f1 = torch.from_numpy(np.load(path1)).float()
            vid1 = self.annotations[self.filenames[idx]]
            duration1 = vid1['video_duration']
            boundaries1 = []
            for annotations in vid1['substages_timestamps']:
                tmp = torch.zeros(FEATURE_LEN,)
                boundary = []
                for pt in annotations:
                    boundary.append(pt)
                boundary.sort()
                boundary = [int(FEATURE_LEN*(i/duration1)) if i/duration1 < 1 else FEATURE_LEN-1 for i in boundary]
                boundary = sorted(list(set(boundary)))
                tmp[boundary] = 1.
                boundaries1.append(tmp)
            if len(boundaries1) > 5:
                boundaries1 = random.sample(boundaries1, 5)
            elif len(boundaries1) < 5:
                for _ in range(5-len(boundaries1)):
                    dummy = boundaries1[0]
                    boundaries1.append(dummy)
            assert len(boundaries1) == 5
            hard_boundaries1 = torch.stack(boundaries1)
            tmp = random.randint(0,10000)
            if tmp < 10000*GLUE_PROB:
                rand_idx = random.randint(0, self.__len__()-1)
                glue_point = random.randint(FEATURE_LEN//4, (FEATURE_LEN//4)*3)
                while True:
                    try:
                        path2 = os.path.join(DATA_PATH_2, self.filenames[rand_idx])
                        f2 = torch.from_numpy(np.load(path2)).float()
                        break
                    except:
                        rand_idx = random.randint(0, self.__len__()-1)
                f2 = torch.from_numpy(np.load(path2)).float()
                vid2 = self.annotations[self.filenames[rand_idx]]
                duration2 = vid2['video_duration']
                boundaries2 = []
                for annotations in vid2['substages_timestamps']:
                    tmp = torch.zeros(FEATURE_LEN,)
                    boundary = []
                    for pt in annotations:
                        boundary.append(pt)
                    boundary.sort()
                    boundary = [int(FEATURE_LEN*(i/duration2)) if i/duration2 < 1 else FEATURE_LEN-1 for i in boundary]
                    boundary = sorted(list(set(boundary)))
                    tmp[boundary] = 1.
                    boundaries2.append(tmp)
                if len(boundaries2) > 5:
                    boundaries2 = random.sample(boundaries2, 5)
                elif len(boundaries2) < 5:
                    for _ in range(5-len(boundaries2)):
                        dummy = boundaries2[0]
                        boundaries2.append(dummy)
                assert len(boundaries2) == 5
                hard_boundaries2 = torch.stack(boundaries2)
                #glueing
                f = torch.cat((f1[:glue_point], f2[glue_point:]), dim=0)
                hard_boundaries = torch.cat((hard_boundaries1[:, :glue_point], hard_boundaries2[:, glue_point:]), dim=1)
                hard_boundaries[:, glue_point] = 1.
            else:
                f = f1
                hard_boundaries = hard_boundaries1
            
            soft_boundaries = 0
            return f, hard_boundaries, soft_boundaries




        path1 = os.path.join(DATA_PATH, self.filenames[idx])
        f1 = torch.from_numpy(np.load(path1)).float()
        vid1 = self.annotations[self.filenames[idx]]
        boundaries1 = []
        for annotations in vid1['substages_timestamps']:
            tmp = torch.zeros(FEATURE_LEN,)
            boundary = []
            for pt in annotations:
                boundary.append(pt)
            boundary.sort()
            boundary = [int(i/TIME_UNIT) if int(i/TIME_UNIT) < FEATURE_LEN else FEATURE_LEN-1 for i in boundary]
            boundary = sorted(list(set(boundary)))
            tmp[boundary] = 1.
            boundaries1.append(tmp)
        if len(boundaries1) > 5:
            boundaries1 = random.sample(boundaries1, 5)
        elif len(boundaries1) < 5:
            for _ in range(5-len(boundaries1)):
                dummy = boundaries1[0]
                boundaries1.append(dummy)
        assert len(boundaries1) == 5
        hard_boundaries1 = torch.stack(boundaries1)

        rand_idx = random.randint(0, self.__len__()-1)
        glue_point = random.randint(FEATURE_LEN//4, (FEATURE_LEN//4)*3)
        path2 = os.path.join(DATA_PATH, self.filenames[rand_idx])
        f2 = torch.from_numpy(np.load(path2)).float()
        vid2 = self.annotations[self.filenames[rand_idx]]
        boundaries2 = []
        for annotations in vid2['substages_timestamps']:
            tmp = torch.zeros(FEATURE_LEN,)
            boundary = []
            for pt in annotations:
                boundary.append(pt)
            boundary.sort()
            boundary = [int(i/TIME_UNIT) if int(i/TIME_UNIT) < FEATURE_LEN else FEATURE_LEN-1 for i in boundary]
            boundary = sorted(list(set(boundary)))
            tmp[boundary] = 1.
            boundaries2.append(tmp)
        if len(boundaries2) > 5:
            boundaries2 = random.sample(boundaries2, 5)
        elif len(boundaries2) < 5:
            for _ in range(5-len(boundaries2)):
                dummy = boundaries2[0]
                boundaries2.append(dummy)
        assert len(boundaries2) == 5
        hard_boundaries2 = torch.stack(boundaries2)
        #glueing
        tmp = random.randint(0,10000)
        if tmp < 10000*GLUE_PROB:
            f = torch.cat((f1[:glue_point], f2[glue_point:]), dim=0)
            hard_boundaries = torch.cat((hard_boundaries1[:, :glue_point], hard_boundaries2[:, glue_point:]), dim=1)
            hard_boundaries[:, glue_point] = 1.
        else:
            f = f1
            hard_boundaries = hard_boundaries1
        s,k = 1,1
        gaussian_filter = torch.FloatTensor([np.exp(-z*z/(2*s*s))/np.sqrt(2*np.pi*s*s) for z in range(-k,k+1)]).unsqueeze(0).unsqueeze(0)
        gaussian_filter *= 1/torch.max(gaussian_filter)
        # soft_boundaries = nn.functional.conv1d(hard_boundaries.unsqueeze(1),gaussian_filter, padding=k)
        # soft_boundaries = torch.clamp(soft_boundaries.squeeze(), min=0.0, max=1.)
        soft_boundaries = 0
        return f, hard_boundaries, soft_boundaries

class Kinetics_GEBD_validation(Dataset):
    def __init__(self, n_fold):
        with open(ANNOTATION_PATH, 'rb') as f:
            self.annotations = pickle.load(f)
        with open(FILE_LIST, 'rb') as f:
            self.filenames = pickle.load(f)[n_fold]['validation']

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = os.path.join(DATA_PATH, self.filenames[idx])
        duration = self.annotations[self.filenames[idx]]['video_duration']
        f = torch.from_numpy(np.load(path)).float()
        return f, self.filenames[idx], duration

class Kinetics_GEBD_test(Dataset):
    def __init__(self):
        with open(TEST_ANNOTATION_PATH, 'r', encoding='utf-8') as f:
            self.video_durations = json.load(f)
        self.filenames = os.listdir(TEST_DATA_PATH)
            
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = os.path.join(TEST_DATA_PATH, self.filenames[idx])
        duration = self.video_durations[self.filenames[idx]]['video_duration']
        f = torch.from_numpy(np.load(path)).float()
        return f, self.filenames[idx], duration



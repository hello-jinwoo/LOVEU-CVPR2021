import os
import pickle
import json
import random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from config import *

def get_boundaries(video_annotation, interpolation=False):
    '''
    IN: annotation
    OUT: event, shot, whole
    '''
    event_boundaries = []
    for annotations in video_annotation['change_event']:
        tmp = torch.zeros(FEATURE_LEN,)
        boundary = []
        for pt in annotations:
            boundary.append(pt)
        boundary.sort()
        boundary = [int(i/TIME_UNIT) if int(i/TIME_UNIT) < FEATURE_LEN else FEATURE_LEN-1 for i in boundary]
        boundary = sorted(list(set(boundary)))
        tmp[boundary] = 1.
        event_boundaries.append(tmp)
    if len(event_boundaries) > 5:
        event_boundaries = random.sample(event_boundaries, 5)
    elif len(event_boundaries) < 5:
        for _ in range(5-len(event_boundaries)):
            dummy = event_boundaries[0]
            event_boundaries.append(dummy)
    assert len(event_boundaries) == 5
    event_boundaries = torch.stack(event_boundaries)

    shot_boundaries = []
    for annotations in video_annotation['change_shot']:
        tmp = torch.zeros(FEATURE_LEN,)
        boundary = []
        for pt in annotations:
            boundary.append(pt)
        boundary.sort()
        if interpolation:
            duration = video_annotation['video_duration']
            boundary = [int(FEATURE_LEN*(i/duration)) if i/duration < 1 else FEATURE_LEN-1 for i in boundary]
        else:
            boundary = [int(i/TIME_UNIT) if int(i/TIME_UNIT) < FEATURE_LEN else FEATURE_LEN-1 for i in boundary]
        boundary = sorted(list(set(boundary)))
        tmp[boundary] = 1.
        shot_boundaries.append(tmp)
    if len(shot_boundaries) > 5:
        shot_boundaries = random.sample(shot_boundaries, 5)
    elif len(shot_boundaries) < 5:
        for _ in range(5-len(shot_boundaries)):
            dummy = shot_boundaries[0]
            shot_boundaries.append(dummy)
    assert len(shot_boundaries) == 5
    shot_boundaries = torch.stack(shot_boundaries)

    whole_boundaries = []
    for annotations in video_annotation['substages_timestamps']:
        tmp = torch.zeros(FEATURE_LEN,)
        boundary = []
        for pt in annotations:
            boundary.append(pt)
        boundary.sort()
        boundary = [int(i/TIME_UNIT) if int(i/TIME_UNIT) < FEATURE_LEN else FEATURE_LEN-1 for i in boundary]
        boundary = sorted(list(set(boundary)))
        tmp[boundary] = 1.
        whole_boundaries.append(tmp)
    if len(whole_boundaries) > 5:
        whole_boundaries = random.sample(whole_boundaries, 5)
    elif len(whole_boundaries) < 5:
        for _ in range(5-len(whole_boundaries)):
            dummy = whole_boundaries[0]
            whole_boundaries.append(dummy)
    assert len(whole_boundaries) == 5
    whole_boundaries = torch.stack(whole_boundaries)
    
    return event_boundaries, shot_boundaries, whole_boundaries

class Kinetics_GEBD_train(Dataset):
    def __init__(self, n_fold):
        with open(ANNOTATION_PATH, 'rb') as f:
            self.annotations = pickle.load(f)
        with open(FILE_LIST, 'rb') as f:
            self.filenames = pickle.load(f)[n_fold]['train']

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        tmp_interpolation = random.randint(0,10000)

        if tmp_interpolation < 10000*INTERPOLATION_PROB:
            path1 = os.path.join(DATA_PATH_2, self.filenames[idx])
            f1 = torch.from_numpy(np.load(path1)).float()
            vid1_annotation = self.annotations[self.filenames[idx]]
            event1, shot1, whole1 = get_boundaries(vid1_annotation, interpolation=True)
        else:
            path1 = os.path.join(DATA_PATH, self.filenames[idx])
            f1 = torch.from_numpy(np.load(path1)).float()
            vid1_annotation = self.annotations[self.filenames[idx]]
            event1, shot1, whole1 = get_boundaries(vid1_annotation)

        
        
        #glueing
        tmp = random.randint(0,10000)
        if tmp < 10000*GLUE_PROB:
            rand_idx = random.randint(0, self.__len__()-1)
            glue_point = random.randint(FEATURE_LEN // 4, (FEATURE_LEN // 4) * 3)
            
            if tmp_interpolation < 10000*INTERPOLATION_PROB:
                path2 = os.path.join(DATA_PATH_2, self.filenames[rand_idx])
                f2 = torch.from_numpy(np.load(path2)).float()
                vid2_annotation = self.annotations[self.filenames[rand_idx]]
                event2, shot2, whole2 = get_boundaries(vid2_annotation, interpolation=True)
            else:
                path2 = os.path.join(DATA_PATH, self.filenames[rand_idx])
                f2 = torch.from_numpy(np.load(path2)).float()
                vid2_annotation = self.annotations[self.filenames[rand_idx]]
                event2, shot2, whole2 = get_boundaries(vid2_annotation)

            f = torch.cat((f1[:glue_point], f2[glue_point:]), dim=0)
            event_boundaries = torch.cat((event1[:, :glue_point], event2[:, glue_point:]), dim=1)
            shot_boundaries = torch.cat((shot1[:, :glue_point], shot2[:, glue_point:]), dim=1)
            shot_boundaries[:, glue_point] = 1.
            whole_boundaries = torch.cat((whole1[:, :glue_point], whole2[:, glue_point:]), dim=1)
            whole_boundaries[:, glue_point] = 1.
        else:
            f = f1
            event_boundaries = event1
            shot_boundaries = shot1
            whole_boundaries = whole1
        return f, event_boundaries, shot_boundaries, whole_boundaries

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



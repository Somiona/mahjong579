import copy
import os
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .tenhou import TenhouData


def process_data(one_batch, label_trans=None):
    features = []
    labels = []
    for f, lb in one_batch:
        features.append(f)
        labels.append(lb)
    features = torch.from_numpy(np.array(features)).float()
    labels = torch.from_numpy(np.array(labels))
    if callable(label_trans):
        labels = label_trans(labels)
    return features, labels


def process_reward_data(one_batch):
    features = []
    labels = []
    for f, lb in one_batch:
        features.append(torch.from_numpy(f).float())
        labels.append(lb)
    features = pad_sequence(features, batch_first=True)
    labels = torch.tensor(labels).float().unsqueeze(-1) / 250
    return features, labels


class TenhouDataset(object):
    def __init__(self, data_dir, batch_size, mode='discard', target_length=1):
        self.data_dir = data_dir
        self.used_data = []
        self.batch_size = batch_size
        data_files = os.listdir(data_dir)
        self.data_files =[f for f in data_files if f.endswith('.xml')]
        random.shuffle(self.data_files)
        self.data_buffer = []
        self.func = f'parse_{mode}_data'
        self.target = slice(0, target_length)

    def reset(self):
        self.data_files = copy.copy(self.used_data)
        random.shuffle(self.data_files)
        self.used_data.clear()

    def __len__(self):
        return len(self.data_files)

    def update_buffer(self):
        if len(self.data_files) == 0:
            return False
        data_file = self.data_files.pop()
        self.used_data.append(data_file)
        playback = TenhouData(os.path.join(self.data_dir, data_file))
        targets = playback.get_rank()[self.target]
        for target in targets:
            features, labels = playback.__getattribute__(self.func)(target=target)
            if isinstance(features, list):
                data = list(zip(features, labels))
                random.shuffle(data)
                self.data_buffer.extend(data)
            else:
                self.data_buffer.append((features, labels))
        return True

    def __call__(self):
        while len(self.data_buffer) < self.batch_size:
            success = self.update_buffer()
            if not success:
                break
        data, self.data_buffer = self.data_buffer[:self.batch_size], self.data_buffer[self.batch_size:]
        return data

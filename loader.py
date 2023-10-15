####################################################################################################
# HELP: hardware-adaptive efficient latency prediction for nas via meta-learning, NeurIPS 2021
# Hayeon Lee, Sewoong Lee, Song Chong, Sung Ju Hwang
# github: https://github.com/HayeonLee/HELP, email: hayeon926@kaist.ac.kr
####################################################################################################
import os

import numpy as np
import torch

from utils import *


class Data:
    def __init__(self,
                mode,
                data_path,
                meta_train_devices,
                meta_valid_devices,
                meta_test_devices,
                num_inner_tasks,
                num_meta_train_sample,
                num_sample,
                num_query,
                num_query_meta_train_task=200,
                remove_outlier=True):
        self.mode = mode
        self.data_path = data_path
        self.meta_train_devices = meta_train_devices
        self.meta_valid_devices = meta_valid_devices
        self.meta_test_devices = meta_test_devices
        self.num_inner_tasks = num_inner_tasks
        self.num_meta_train_sample = num_meta_train_sample
        self.num_sample = num_sample
        self.num_query = num_query
        self.num_query_meta_train_task = num_query_meta_train_task
        self.remove_outlier = remove_outlier

        self.load_archs()

        self.train_idx ={}
        self.valid_idx = {}
        self.latency = {}
        self.norm_latency = {}
        nts = self.num_meta_train_sample
        for device in meta_train_devices + meta_valid_devices + meta_test_devices:
            self.latency[device] = torch.FloatTensor(
                torch.load(os.path.join(data_path, 'latency', f'{device}.pt')))
            train_idx = torch.arange(len(self.archs))[:nts]
            valid_idx = torch.arange(len(self.archs))[nts:nts+self.num_query]

            if self.remove_outlier:
                self.train_idx[device] = train_idx[
                    np.argsort(self.latency[device][train_idx])[
                        int(len(train_idx)*0.1):int(len(train_idx)*0.9)]]
                self.valid_idx[device] = valid_idx[
                    np.argsort(self.latency[device][valid_idx])[
                        int(len(valid_idx)*0.1):int(len(valid_idx)*0.9)]]

            self.norm_latency[device] = normalization(
                                        latency=self.latency[device],
                                        index = self.train_idx[device]
                                        )
        # load index set of reference architectures
        self.hw_emb_idx = torch.load(
            os.path.join(data_path, f'{self.head_or_backbone}_hardware_embedding_index.pt'))

        if self.mode == 'nas':
            self.max_lat_idx, self.min_lat_idx = get_minmax_latency_index(
                meta_train_devices + meta_valid_devices, self.train_idx, self.latency)
            self.nas_norm_latency = {}
            for device in meta_train_devices + meta_valid_devices + meta_test_devices:
                self.nas_norm_latency[device] = normalization(
                    latency=self.latency[device],
                    index = torch.tensor([self.max_lat_idx, self.min_lat_idx] + self.hw_emb_idx))

        print('==> load data ...')


    def load_archs(self):
        devices = self.meta_train_devices + self.meta_valid_devices + self.meta_test_devices
        print(devices)
        head_or_backbone = devices[0].split('_')[-1] # 'head' or 'backbone'
        if head_or_backbone not in ['head', 'backbone']: raise ValueError('Device 이름은 "_head"나 "_backbone"으로 끝나야 합니다.')

        max_depth = 5 if head_or_backbone == 'backbone' else 7
        self.head_or_backbone = head_or_backbone

        self.archs = [arch_encoding_ofa(arch, max_depth) for arch in
            torch.load(os.path.join(self.data_path, f'{head_or_backbone}_archs.pt'))['arch']]


    def generate_episode(self):
        # metabatch
        episode = []

        # meta-batch
        rand_device_idx = torch.randperm(
                            len(self.meta_train_devices))[:self.num_inner_tasks]
        for t in rand_device_idx:
            # sample devices
            device = self.meta_train_devices[t]
            # hardware embedding
            latency = self.latency[device]
            hw_embed = latency[self.hw_emb_idx]
            hw_embed = normalization(hw_embed, portion=1.0)

            # samples for finetuning & test (query)
            rand_idx = self.train_idx[device][torch.randperm(len(self.train_idx[device]))]
            finetune_idx = rand_idx[:self.num_sample]
            qry_idx = rand_idx[self.num_sample:self.num_sample+self.num_query_meta_train_task]

            x_finetune = torch.stack([self.archs[_] for _ in finetune_idx])
            x_qry = torch.stack([self.archs[_] for _ in qry_idx])

            y_finetune = self.norm_latency[device][finetune_idx].view(-1, 1)
            y_qry = self.norm_latency[device][qry_idx].view(-1, 1)

            episode.append((hw_embed, x_finetune, y_finetune, x_qry, y_qry, device))

        return episode

    def generate_test_tasks(self, split=None):
        if split == 'meta_train':
            device_list = self.meta_train_devices
        elif split == 'meta_valid':
            device_list = self.meta_valid_devices
        elif split == 'meta_test':
            device_list = self.meta_test_devices
        else: NotImplementedError

        tasks = []
        for device in device_list:
            tasks.append(self.get_task(device))
        return tasks

    def get_task(self, device=None, num_sample=None):
        if num_sample == None:
            num_sample = self.num_sample

        latency = self.latency[device]
        # hardware embedding
        hw_embed = latency[self.hw_emb_idx]
        hw_embed = normalization(hw_embed, portion=1.0)

        # samples for finetuing & test (query)
        rand_idx = self.train_idx[device][torch.randperm(len(self.train_idx[device]))]
        finetune_idx = rand_idx[:num_sample]

        x_finetune = torch.stack([self.archs[_] for _ in finetune_idx])
        x_qry = torch.stack([self.archs[_] for _ in self.valid_idx[device]])

        y_finetune = self.norm_latency[device][finetune_idx].view(-1, 1)
        y_qry = self.norm_latency[device][self.valid_idx[device]].view(-1, 1)

        return hw_embed, x_finetune, y_finetune, x_qry, y_qry, device


    def get_nas_task(self, device=None):
        latency = self.latency[device]
        # hardware embedding
        hw_embed = latency[self.hw_emb_idx]
        hw_embed = normalization(hw_embed, portion=1.0)

        # samples for finetuning & test (query)
        finetune_idx = self.hw_emb_idx
        norm_latency = self.nas_norm_latency[device]
        x_finetune = torch.stack([self.archs[_] for _ in finetune_idx])

        y_finetune = norm_latency[finetune_idx].view(-1, 1)
        y_finetune_gt = latency[finetune_idx].view(-1, 1)

        return hw_embed, x_finetune, y_finetune, y_finetune_gt
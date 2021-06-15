'''
SSMix
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np

class NLUDataset(Dataset):
    def __init__(self, args, glue_dataset, label_list, mode='train'):
        self.args = args
        self.mode = mode
        if self.mode == 'train':
            glue_dataset, label_list = self.split_data(glue_dataset, label_list)
        self.input_ids = glue_dataset['input_ids']
        self.attn_mask = glue_dataset['attention_mask']
        self.ttids = glue_dataset['token_type_ids']
        self.label_list = label_list
        self.mode = mode
        self.tokenized_length = self.get_token_length()

    def split_data(self, glue_dataset, label_list):
        train_idxs, _ = train_test_split(np.array(range(len(glue_dataset['input_ids']))))
        train_idxs = np.array(train_idxs)
        dataset = dict()
        for key in ['input_ids', 'attention_mask', 'token_type_ids']:
            dataset[key] = glue_dataset[key][train_idxs, :]
        label_list = np.array(label_list)[train_idxs]
        return dataset, label_list

    def __len__(self):
        length = len(self.label_list)
        if self.mode == 'train':
            if length % 2 == 1:
                length -= 1 # make the dataset size even (In order to conduct pairwise mixup)
        return length

    def __getitem__(self, idx):
        inputs = {'input_ids': self.input_ids[idx],
                  'attention_mask': self.attn_mask[idx],
                  'token_type_ids': self.ttids[idx]}
        return inputs, self.label_list[idx], self.tokenized_length[idx]

    def get_token_length(self):
        return [x.sum() for x in self.attn_mask]

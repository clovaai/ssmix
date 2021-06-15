'''
SSMix
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
import copy
import random

import torch


class UNK:
    def __init__(self, args):
        self.args = args

    def __call__(self, input1, input2, target1, target2, length1, length2, max_len):  # random word replace with [UNK]
        batch_size, n_token = input1['input_ids'].shape
        inputs_aug = copy.deepcopy(input1)
        len1 = length1.clone().detach()
        ratio = torch.ones((batch_size,), device=self.args.device)
        for i in range(batch_size):  # force augmented output length to be no more than max_len
            if len1[i].item() > max_len:
                len1[i] = max_len
                for key in inputs_aug.keys():
                    inputs_aug[key][i][max_len:] = 0
                inputs_aug['input_ids'][i][max_len - 1] = 102
            mix_len = int((len1[i] - 2) * (self.args.unk_winsize / 100.)) or 1
            flip_idx = random.sample(range(1, len1[i] - 1), mix_len)
            inputs_aug['input_ids'][i][flip_idx] = 100  # 100 is unk token in bert.
            ratio[i] = 1  # - (mix_len / (len1[i].item() - 2))
        return inputs_aug, ratio


'''
SSMix
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
import copy
import random
import torch
import torch.nn.functional as F

from .saliency import get_saliency


class SSMix:
    def __init__(self, args):
        self.args = args

    def __call__(self, input1, input2, target1, target2, length1, length2, max_len):
        batch_size = len(length1)

        if self.args.ss_no_saliency:
            if self.args.ss_no_span:
                inputs_aug, ratio = self.ssmix_nosal_nospan(input1, input2, length1, length2, max_len)
            else:
                inputs_aug, ratio = self.ssmix_nosal(input1, input2, length1, length2, max_len)
        else:
            assert not self.args.ss_no_span

            input2_saliency, input2_emb, _ = get_saliency(self.args, input2, target2)

            inputs_aug, ratio = self.ssmix(batch_size, input1, input2,
                                           length1, length2, input2_saliency, target1, max_len)

        return inputs_aug, ratio

    def ssmix(self, batch_size, input1, input2, length1, length2, saliency2, target1, max_len):
        inputs_aug = copy.deepcopy(input1)
        for i in range(batch_size):  # cut off length bigger than max_len ( nli task )
            if length1[i].item() > max_len:
                length1[i] = max_len
                for key in inputs_aug.keys():
                    inputs_aug[key][i][max_len:] = 0
                inputs_aug['input_ids'][i][max_len - 1] = 102
        saliency1, _, _ = get_saliency(self.args, inputs_aug, target1)
        ratio = torch.ones((batch_size,), device=self.args.device)

        for i in range(batch_size):
            l1, l2 = length1[i].item(), length2[i].item()
            limit_len = min(l1, max_len) - 2  # mixup except [CLS] and [SEP]
            mix_size = max(int(limit_len * (self.args.ss_winsize / 100.)), 1)

            if l2 < mix_size:
                ratio[i] = 1
                continue

            saliency1_nopad = saliency1[i, :l1].unsqueeze(0).unsqueeze(0)
            saliency2_nopad = saliency2[i, :l2].unsqueeze(0).unsqueeze(0)

            saliency1_pool = F.avg_pool1d(saliency1_nopad, mix_size, stride=1).squeeze(0).squeeze(0)
            saliency2_pool = F.avg_pool1d(saliency2_nopad, mix_size, stride=1).squeeze(0).squeeze(0)

            # should not select first and last
            saliency1_pool[0], saliency1_pool[-1] = 100, 100
            saliency2_pool[0], saliency2_pool[-1] = -100, -100
            input1_idx = torch.argmin(saliency1_pool)
            input2_idx = torch.argmax(saliency2_pool)
            inputs_aug['input_ids'][i, input1_idx:input1_idx + mix_size] = \
                input2['input_ids'][i, input2_idx:input2_idx + mix_size]

            ratio[i] = 1 - (mix_size / (l1 - 2))

        return inputs_aug, ratio

    def ssmix_nosal(self, input1, input2, length1, length2, max_len):
        inputs_aug = copy.deepcopy(input1)
        ratio = torch.ones((len(length1),), device=self.args.device)

        for idx in range(len(length1)):
            if length1[idx].item() > max_len:
                for key in inputs_aug.keys():
                    inputs_aug[key][idx][max_len:] = 0
                inputs_aug['input_ids'][idx][max_len - 1] = 102  # artificially add EOS token.
            l1, l2 = min(length1[idx].item(), max_len), length2[idx].item()

            if self.args.ss_winsize == -1:
                window_size = random.randrange(0, l1)  # random sampling of window_size
            else:
                # remove EOS & SOS when calculating ratio & window size.
                window_size = int((l1 - 2) *
                                  self.args.ss_winsize / 100.) or 1

            if l2 <= window_size:
                ratio[idx] = 1
                continue

            start_idx = random.randrange(0, l1 - window_size)  # random sampling of starting point
            if (l2 - window_size) < start_idx:  # not enough text for reference.
                ratio[idx] = 1
                continue
            else:
                ref_start_idx = start_idx
            mix_percent = float(window_size) / (l1 - 2)

            for key in input1.keys():
                inputs_aug[key][idx, start_idx:start_idx + window_size] = \
                    input2[key][idx, ref_start_idx:ref_start_idx + window_size]

            ratio[idx] = 1 - mix_percent
        return inputs_aug, ratio

    def ssmix_nosal_nospan(self, input1, input2, length1, length2, max_len):
        batch_size, n_token = input1['input_ids'].shape

        inputs_aug = copy.deepcopy(input1)
        len1 = length1.clone().detach()
        ratio = torch.ones((batch_size,), device=self.args.device)

        for i in range(batch_size): # force augmented output length to be no more than max_len
            if len1[i].item() > max_len:
                len1[i] = max_len
                for key in inputs_aug.keys():
                    inputs_aug[key][i][max_len:] = 0
                inputs_aug['input_ids'][i][max_len - 1] = 102

            mix_len = int((len1[i] - 2) * (self.args.ss_winsize / 100.)) or 1
            if (length2[i] - 2) < mix_len:
                mix_len = length2[i] - 2

            flip_idx = random.sample(range(1, min(len1[i] - 1, length2[i] - 1)), mix_len)
            inputs_aug['input_ids'][i][flip_idx] = input2['input_ids'][i][flip_idx]
            ratio[i] = 1 - (mix_len / (len1[i].item() - 2))

        return inputs_aug, ratio





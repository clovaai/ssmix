'''
SSMix
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
import numpy as np
from transformers import *
from datasets import load_dataset, load_metric, concatenate_datasets
import torch

task_to_keys = {
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "trec": ("text", None),
    "anli": ("premise", "hypothesis"),
}


class PreProcessor:
    def __init__(self, args=None, task_name='sst2', tokenizer=None, max_len=128, seed_num=0):
        set_seed(seed_num)
        self.args = args
        self.task_name = task_name
        self.tokenizer = tokenizer
        if 'trec' in self.task_name:
            self.task_name = 'trec'
            self.datasets = load_dataset('trec')
        elif self.task_name == 'anli':
            self.datasets = load_dataset('anli')
        else: # glue task
            self.datasets = load_dataset("glue", self.task_name)
        self.max_length = max_len

        self.sentence1_key, self.sentence2_key, self.train_dataset, self.eval_dataset, \
        self.test_dataset, self.compute_metrics, self.num_labels, self.eval_key = None, None, None, None, \
                                                                                  None, None, None, None

        self.get_label_info()
        self.preprocess_dataset()
        self.get_metric()

    def get_label_info(self):
        # Labels
        if self.task_name == 'trec':
            if self.args.dataset == 'trec-fine':
                label_list = self.datasets["train"].features["label-fine"].names
            elif self.args.dataset == 'trec-coarse':
                label_list = self.datasets["train"].features["label-coarse"].names
            self.num_labels = len(label_list)   # 6
        elif self.args.dataset == 'anli':
            label_list = self.datasets["train_r1"].features["label"].names
            self.num_labels = len(label_list)
        else:
            label_list = self.datasets["train"].features["label"].names
            self.num_labels = len(label_list)

    def preprocess_dataset(self):
        def preprocess_function(examples):
            # Tokenize the texts
            args = (
                (examples[self.sentence1_key],) if self.sentence2_key is None else (examples[self.sentence1_key],
                                                                                    examples[self.sentence2_key])
            )
            result = self.tokenizer(*args, padding='max_length', max_length=self.max_length, truncation=True)
            return result

        self.sentence1_key, self.sentence2_key = task_to_keys[self.task_name]
        self.datasets = self.datasets.map(preprocess_function, batched=True, load_from_cache_file=True)
        self.datasets.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])
        if self.task_name == 'anli':
            self.train_dataset = concatenate_datasets([self.datasets['train_r1'], self.datasets['train_r2'], self.datasets['train_r3']])
        else:
            self.train_dataset = self.datasets['train']

        if self.task_name == 'anli':
            self.eval_dataset = {
                'test_r1': self.datasets['test_r1'],
                'test_r2': self.datasets['test_r2'],
                'test_r3': self.datasets['test_r3'],
                'val_r1': self.datasets['dev_r1'],
                'val_r2': self.datasets['dev_r2'],
                'val_r3': self.datasets['dev_r3'],
            }
        else:
            if self.task_name == 'mnli':
                self.eval_key = 'validation_matched'
            elif self.task_name == 'trec':
                self.eval_key = 'test'
            else:
                self.eval_key = 'validation'
            self.eval_dataset = self.datasets[self.eval_key]

    def get_metric(self):
        # Get the metric function
        if self.task_name == 'trec' or self.task_name == 'anli':
            self.compute_metrics = None
            return
        metric = load_metric("glue", self.task_name)

        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            if self.task_name is not None:
                result = metric.compute(predictions=preds, references=p.label_ids)
                if len(result) > 1:
                    result["combined_score"] = np.mean(list(result.values())).item()
                return result
            else:
                return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

        self.compute_metrics = compute_metrics

    def get_accuracy(self, preds, label_ids):
        if self.task_name == 'trec' or self.task_name == 'anli':
            predicted = torch.argmax(preds, dim=1)
            correct = (predicted == label_ids).sum()
            total_sample = len(label_ids)
            return float(correct) / total_sample
        return self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))

'''
SSMix
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
from .dataset import *
from .preprocess import *


def get_data(args, max_seq_len=128):
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)

    # Build the dataset class for each set
    processor = PreProcessor(args=args, task_name=args.dataset, tokenizer=tokenizer, max_len=max_seq_len, seed_num=args.seed)
    if args.dataset == 'trec-fine':
        label_name = 'label-fine'
    elif args.dataset == 'trec-coarse':
        label_name = 'label-coarse'
    else:
        label_name = 'label'

    train_labeled_dataset = NLUDataset(args, processor.train_dataset,
                                       processor.train_dataset[label_name], mode='train')
    if args.dataset == 'anli':
        test_dataset = {
                'test_r1': NLUDataset(args, processor.eval_dataset['test_r1'], processor.eval_dataset['test_r1'][label_name], mode='eval'),
                'test_r2': NLUDataset(args, processor.eval_dataset['test_r2'], processor.eval_dataset['test_r2'][label_name], mode='eval'),
                'test_r3': NLUDataset(args, processor.eval_dataset['test_r3'], processor.eval_dataset['test_r3'][label_name], mode='eval'),
                'val_r1': NLUDataset(args, processor.eval_dataset['val_r1'], processor.eval_dataset['val_r1'][label_name], mode='eval'),
                'val_r2': NLUDataset(args, processor.eval_dataset['val_r2'], processor.eval_dataset['val_r2'][label_name], mode='eval'),
                'val_r3': NLUDataset(args, processor.eval_dataset['val_r3'], processor.eval_dataset['val_r3'][label_name], mode='eval')
            }
        test_len = len(test_dataset['test_r1']) + len(test_dataset['test_r2']) + len(test_dataset['test_r3'])
        val_len = len(test_dataset['val_r1']) + len(test_dataset['val_r3']) + len(test_dataset['val_r3'])
        print(f"ANLI test_len: {test_len}, val_len: {val_len}")
    else:
        test_dataset = NLUDataset(args, processor.eval_dataset, processor.eval_dataset[label_name], mode='eval')

    n_labels = processor.num_labels
    print(f"| Number of Labeled Samples : {len(train_labeled_dataset)} \t "
          f"Number of Test Samples : {len(test_dataset)} \t n_labels : {n_labels}")

    return train_labeled_dataset, test_dataset, n_labels, tokenizer, processor.get_accuracy

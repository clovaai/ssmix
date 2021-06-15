'''
SSMix
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AdamW, get_linear_schedule_with_warmup, set_seed

from classification_model import ClassificationModel
from trainer import Trainer
from read_data import get_data


def run_train(args):
    # load data
    loader = dict()

    train_labeled_set, test_set, n_labels, tokenizer, eval_fn = get_data(args)
    args.eval_fn = eval_fn

    loader['labeled_trainloader'] = DataLoader(dataset=train_labeled_set,
                                               batch_size=args.batch_size, shuffle=True)
    if args.dataset == 'anli':
        loader['test_loader'] = dict()
        for key in ["test_r1", 'test_r2', 'test_r3', 'val_r1', 'val_r2', 'val_r3']:
            loader['test_loader'][key] = DataLoader(dataset=test_set[key], batch_size=1024, shuffle=False)
    else:
        loader['test_loader'] = DataLoader(dataset=test_set, batch_size=1024, shuffle=False)

    print("total number of labels: ", n_labels)

    # Configure epoch/step number
    loader_length = len(loader['labeled_trainloader'])
    args.steps = 0
    if loader_length < 500:
        args.checkpoint = loader_length
    else:
        args.checkpoint = 500
    print(f"{args.dataset}: loader-length: {loader_length}, "
          f"total of {(5 * loader_length) // args.checkpoint} checkpoints for aug(*3/5 for normal)")

    # load model
    model = ClassificationModel(pretrained_model=args.pretrained_model, num_labels=n_labels).to(args.device)
    model = nn.DataParallel(model)

    # Load pretrained models for augmentation
    if args.aug_mode == 'normal':
        args.epochs = 3
        assert args.optimizer_lr == 5e-05
    else:
        checkpoint_name = f"{args.dataset}-{args.seed}"
        if 'trec' in args.dataset:
            if args.dataset == 'trec-fine':
                checkpoint_name = f"f-trec-{args.seed}"
            else:
                checkpoint_name = f"c-trec-{args.seed}"
        elif args.dataset == 'anli':
            checkpoint_name = f"{args.dataset}-{args.anli_round}-{args.seed}"

        checkpoint_name = f'{args.checkpoint_path}/{checkpoint_name}/best.pt'

        model.load_state_dict(torch.load(checkpoint_name, map_location=args.device))
        args.epochs = 5
    print(args)

    # Configure optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 1e-4  # 10^-4 good at mixup paper
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.optimizer_lr, eps=1e-8)

    # warmup for 10% of total training step.
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=(loader_length * args.epochs) // 10,
                                                num_training_steps=loader_length * args.epochs)
    criterion = nn.CrossEntropyLoss(reduction="none")

    trainer = Trainer(args=args, model=model, optimizer=optimizer, criterion=criterion, loader=loader,
                      n_labels=n_labels, tokenizer=tokenizer, scheduler=scheduler)
    trainer.run_train()


def parse_argument():
    parser = argparse.ArgumentParser(description='train classification model')
    parser.add_argument('--pretrained_model', type=str, default='bert-base-uncased', help='pretrained model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')

    parser.add_argument('--verbose', action='store_true', help='description T/F for printing out the logs')
    parser.add_argument('--verbose_level', type=int, default=2000,
                        help='description level for printing out the logs - print every n batch')
    parser.add_argument('--verbose_show_augment_example', action='store_true',
                        help='Print out examples of augmented text at first epoch, '
                             'and also print out initial test accuracy')
    parser.add_argument('--seed', type=int, help='Set seed number')
    parser.add_argument('--optimizer_lr', type=float, default=5e-05, help='Set learning rate for optimizer')
    parser.add_argument('--naive_augment', action='store_true', help='Augment without original data')
    parser.add_argument('--dataset', type=str, default='trec', help='Dataset to use')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/',
                        help='Directory path to save checkpoint')
    parser.add_argument('--anli_round', type=int, default=1, choices=[1, 2, 3],
                        help='dataset to load for ANLI round.')

    # Training mode. AUG_MODE must be one of following modes
    subparsers = parser.add_subparsers(title='augmentation', description='Augmentation mode', dest='aug_mode')

    # NORMAL (No augmentation)
    subparsers.add_parser('normal')
    subparsers.default = 'normal'

    # SSMIX
    sp_ss = subparsers.add_parser('ssmix')
    sp_ss.add_argument('--ss_winsize', type=int, default=10,
                       help='Percent of window size. 10 means 10% for augmentation')
    sp_ss.add_argument('--ss_no_saliency', action='store_true',
                       help='Excluding saliency constraint in SSMix')
    sp_ss.add_argument('--ss_no_span', action='store_true',
                       help='Excluding span constraint in SSMix')

    # TMIX
    sp_hidden = subparsers.add_parser('tmix')
    sp_hidden.add_argument('--hidden_alpha', type=float, default=0.2,
                           help='mixup alpha value for l=np.random.beta(alpha, alpha) when getting lambda probability')

    # EMBEDMIX
    sp_embed = subparsers.add_parser('embedmix')
    sp_embed.add_argument('--embed_alpha', type=float, default=0.2,
                          help='mixup alpha value for l=np.random.beta(alpha, alpha) when getting lambda probability')

    # UNK
    sp_unk = subparsers.add_parser('unk')
    sp_unk.add_argument('--unk_winsize', type=float, default=10,
                        help='Percent of window size. 10 means 10% for augmentation')

    args, _ = parser.parse_known_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset configuration
    args.nli_dataset = args.dataset in ['mnli', 'mrpc', 'qqp', 'qnli', 'rte', 'anli']

    return args


def main():
    args = parse_argument()
    set_seed(args.seed)
    run_train(args)


if __name__ == '__main__':
    print(f"\nTrain pipeline start \n"
          f"CUDA available: {torch.cuda.is_available()}, number of GPU: {torch.cuda.device_count()}")
    main()

'''
SSMix
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
import torch

def get_saliency(args, input, target):
    batch_size = target.shape[0]

    # Saliency
    for key in input.keys():
        if input[key] is not None and len(input[key].size()) < 2:
            input[key] = input[key].unsqueeze(0)
    model = args.model.to(args.device)
    model.train()

    output = model.module.bert(inputs=input, trace_grad=True)
    logit = output[0]
    embedding = output[-1]
    args.optimizer.zero_grad()

    loss = torch.mean(args.criterion(logit, target.to(args.device)))
    loss.backward()

    unary = torch.sqrt(torch.mean(embedding.grad ** 2, dim=2))
    unary = unary / unary.view(batch_size, -1).max(1)[0].view(batch_size, 1)
    return unary, embedding, target

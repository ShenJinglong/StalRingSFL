
import sys

sys.path.append("..")
import logging

import numpy as np
import torch

import wandb
from utils.data_utils import DatasetManager
from utils.model_utils import aggregate_model, construct_model, eval_model


def train(config, device):

    logging.info(f"@ vanilla_fl_drop [{device}]")
    dataset_manager = DatasetManager(config.dataset_name, "./datasets", config.block_num, config.batch_size)
    if config.dataset_type == "iid":
        trainloaders = dataset_manager.get_iid_loaders(config.num_worker)
    elif config.dataset_type == "noniid":
        trainloaders = dataset_manager.get_noniid_loaders(config.num_worker)
    else:
        raise ValueError(f"Unrecognized dataset type: `{config.dataset_type}`")
    testloader = dataset_manager.get_test_loader()

    global_model = construct_model(config.model_type).to(device)
    local_models = [construct_model(config.model_type).to(device) for _ in range(len(trainloaders))]
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optims = [torch.optim.SGD(model.parameters(), config.lr) for model in local_models]

    for round in range(config.global_round):
        logging.info(f"round: {round}")

        part_index = sorted(np.random.choice(len(local_models), (len(local_models)-2,), replace=False))
        logging.info(f"part_index: {part_index}")

        round_local_models = [local_models[ii] for ii in part_index]
        round_trainloaders = [trainloaders[ii] for ii in part_index]
        round_optims = [optims[ii] for ii in part_index]

        [model.load_state_dict(global_model.state_dict()) for model in round_local_models]

        for i, (model, trainloader, optim) in enumerate(zip(round_local_models, round_trainloaders, round_optims)):
            logging.info(f"client {i}")
            for epoch in range(config.local_epoch):
                logging.info(f"epoch: {epoch}")
                for batchdata in trainloader:
                    optim.zero_grad()
                    inputs = batchdata[0].to(device)
                    labels = batchdata[1].to(device)
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    loss.backward()
                    optim.step()

        global_model = aggregate_model(round_local_models, [1./len(round_local_models)]*len(round_local_models))

        acc = eval_model(global_model, testloader)
        logging.info(f"round: {round} | acc: {acc}")
        wandb.log({
            "round": round,
            "acc": acc,
        })

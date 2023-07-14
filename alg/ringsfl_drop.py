
import sys

sys.path.append("..")
import logging

import numpy as np
import torch

import wandb
from utils.data_utils import DatasetManager
from utils.model_utils import (aggregate_model, construct_model, eval_model,
                               get_optim_params)


def train(config, device):

    logging.info(f"@ ringsfl_drop [{device}]")
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
    prop_lens = [int(prop_len) for prop_len in config.prop_lens.split(":")]
    optims = [torch.optim.SGD(optim_params, config.lr) for optim_params in get_optim_params(local_models, prop_lens)]

    for round in range(config.global_round):
        logging.info(f"round: {round}")

        part_index = sorted(np.random.choice(len(local_models), (len(local_models)-2,), replace=False))
        logging.info(f"part_index: {part_index}")

        round_local_models = [local_models[ii] for ii in part_index]
        round_trainloaders = [trainloaders[ii] for ii in part_index]
        round_optims = [optims[ii] for ii in part_index]
        
        if 0 not in part_index:
            round_prop_lens = [4,3,3]
        else:
            round_prop_lens = [8,1,1]

        [model.load_state_dict(global_model.state_dict()) for model in round_local_models]

        for epoch in range(config.local_epoch):
            logging.info(f"epoch: {epoch}")
            for one_batch_datas in zip(*round_trainloaders):
                [optim.zero_grad() for optim in round_optims]

                for i, one_batch_data in enumerate(one_batch_datas):
                    inputs = one_batch_data[0].to(device)
                    labels = one_batch_data[1].to(device)

                    start = 0
                    for j in range(i, i + len(round_local_models)):
                        index = j % len(round_local_models)
                        inputs = round_local_models[index](inputs, start=start, stop=start+round_prop_lens[index])
                        start += round_prop_lens[index]
                    
                    loss = loss_fn(inputs, labels)
                    loss.backward()
                
                [optim.step() for optim in round_optims]

        global_model = aggregate_model(round_local_models, [1./len(round_local_models)]*len(round_local_models))

        # if round == config.drop:
        #     logging.info("Drop last client ...")
        #     del local_models[-1]
        #     del optims[-1]
        #     del trainloaders[-1]
        #     prop_lens[0] += prop_lens[-1]
        #     del prop_lens[-1]

        acc = eval_model(global_model, testloader)
        logging.info(f"round: {round} | acc: {acc}")
        wandb.log({
            "round": round,
            "acc": acc,
        })



import sys

sys.path.append("..")
import logging

import torch

import wandb
from utils.data_utils import DatasetManager
from utils.model_utils import (aggregate_model, construct_model, eval_model,
                               get_optim_params)


def train(config, device):

    logging.info(f"@ ringsfl_subring [{device}]")
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

    trainloaders_g1, trainloaders_g2 = trainloaders[:3], trainloaders[3:]
    local_models_g1, local_models_g2 = local_models[:3], local_models[3:]
    optims_g1, optims_g2 = optims[:3], optims[3:]
    prop_lens_g1, prop_lens_g2 = [8, 1, 1], [8, 1, 1]

    for round in range(config.global_round):
        logging.info(f"round: {round}")
        [model.load_state_dict(global_model.state_dict()) for model in local_models]

        for epoch in range(config.local_epoch):
            logging.info(f"epoch: {epoch}")

            for one_batch_datas in zip(*trainloaders_g1):
                [optim.zero_grad() for optim in optims_g1]

                for i, one_batch_data in enumerate(one_batch_datas):
                    inputs = one_batch_data[0].to(device)
                    labels = one_batch_data[1].to(device)

                    start = 0
                    for j in range(i, i + len(local_models_g1)):
                        index = j % len(local_models_g1)
                        inputs = local_models_g1[index](inputs, start=start, stop=start+prop_lens_g1[index])
                        start += prop_lens_g1[index]
                    
                    loss = loss_fn(inputs, labels)
                    loss.backward()
                
                [optim.step() for optim in optims_g1]
            
            for one_batch_datas in zip(*trainloaders_g2):
                [optim.zero_grad() for optim in optims_g2]

                for i, one_batch_data in enumerate(one_batch_datas):
                    inputs = one_batch_data[0].to(device)
                    labels = one_batch_data[1].to(device)

                    start = 0
                    for j in range(i, i + len(local_models_g2)):
                        index = j % len(local_models_g2)
                        inputs = local_models_g2[index](inputs, start=start, stop=start+prop_lens_g2[index])
                        start += prop_lens_g2[index]
                    
                    loss = loss_fn(inputs, labels)
                    loss.backward()
                
                [optim.step() for optim in optims_g2]

        global_model = aggregate_model(local_models, [1/len(local_models)]*len(local_models))

        acc = eval_model(global_model, testloader)
        logging.info(f"round: {round} | acc: {acc}")
        wandb.log({
            "round": round,
            "acc": acc,
        })


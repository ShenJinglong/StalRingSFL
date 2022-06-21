
import sys
sys.path.append("..")
import torch
import logging
import wandb
import copy

from utils.data_utils import DatasetManager
from utils.model_utils import aggregate_model, eval_splited_model, construct_model, ratio_model_grad

def train(config, device):

    logging.info(f"@ splitfed [{device}]")
    dataset_manager = DatasetManager(config.dataset_name, "./datasets", config.block_num, config.batch_size)
    if config.dataset_type == "iid":
        trainloaders = dataset_manager.get_iid_loaders(config.num_worker)
    elif config.dataset_type == "noniid":
        trainloaders = dataset_manager.get_noniid_loaders(config.num_worker)
    else:
        raise ValueError(f"Unrecognized dataset type: `{config.dataset_type}`")
    testloader = dataset_manager.get_test_loader()

    global_model = construct_model(config.model_type).to(device)
    client_globalmodel = global_model.get_splited_module(config.cut_point)[0]
    server_globalmodel = global_model.get_splited_module(config.cut_point)[1]
    client_localmodels = [copy.deepcopy(client_globalmodel) for _ in range(config.num_worker)]
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    server_optim = torch.optim.SGD(server_globalmodel.parameters(), config.lr)
    client_optims = [torch.optim.SGD(client_localmodel.parameters(), config.lr) for client_localmodel in client_localmodels]
    model_size = sum(p.numel() for p in client_globalmodel.parameters() if p.requires_grad)
    round_comm = 0
    total_comm = 0

    for round in range(config.global_round):
        logging.info(f"round: {round}")
        round_comm = 0
        [client_localmodel.load_state_dict(client_globalmodel.state_dict()) for client_localmodel in client_localmodels]
        round_comm += model_size * len(client_localmodels)

        for epoch in range(config.local_epoch):
            logging.info(f"epoch: {epoch}")
            for one_batch_datas in zip(*trainloaders):
                inputs = [one_batch_data[0].to(device) for one_batch_data in one_batch_datas]
                labels = [one_batch_data[1].to(device) for one_batch_data in one_batch_datas]
                [client_optim.zero_grad() for client_optim in client_optims]
                server_optim.zero_grad()

                feature_maps = [client_localmodel(sample) for client_localmodel, sample in zip(client_localmodels, inputs)]
                round_comm += sum(feature_map.numel() for feature_map in feature_maps)*2
                outputs = [server_globalmodel(feature_map) for feature_map in feature_maps]
                losses = [loss_fn(output, label) for output, label in zip(outputs, labels)]
                [loss.backward() for loss in losses]
                ratio_model_grad(server_globalmodel, 1/config.num_worker)

                [client_optim.step() for client_optim in client_optims]
                server_optim.step()

        client_globalmodel = aggregate_model(client_localmodels, [1/len(client_localmodels)]*len(client_localmodels))
        round_comm += model_size * len(client_localmodels)
        total_comm += round_comm

        acc = eval_splited_model(client_globalmodel, server_globalmodel, testloader)
        logging.info(f"acc: {acc} | round_comm: {round_comm} | total_comm: {total_comm}")
        wandb.log({
            "round": round,
            "acc": acc,
            "round_comm": round_comm,
            "total_comm": total_comm
        })

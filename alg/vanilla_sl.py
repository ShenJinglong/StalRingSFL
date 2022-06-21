
import sys
sys.path.append("..")
import torch
import logging
import wandb

from utils.data_utils import DatasetManager
from utils.model_utils import eval_splited_model, construct_model

def train(config, device):

    logging.info(f"@ vanilla_sl [{device}]")
    dataset_manager = DatasetManager(config.dataset_name, "./datasets", config.block_num, config.batch_size)
    if config.dataset_type == "iid":
        trainloaders = dataset_manager.get_iid_loaders(config.num_worker)
    elif config.dataset_type == "noniid":
        trainloaders = dataset_manager.get_noniid_loaders(config.num_worker)
    else:
        raise ValueError(f"Unrecognized dataset type: `{config.dataset_type}`")
    trainloaders_iter = [iter(trainloader) for trainloader in trainloaders]
    testloader = dataset_manager.get_test_loader()

    global_model = construct_model(config.model_type).to(device)
    client_model = global_model.get_splited_module(config.cut_point)[0]
    server_model = global_model.get_splited_module(config.cut_point)[1]
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    server_optim = torch.optim.SGD(server_model.parameters(), config.lr)
    client_optim = torch.optim.SGD(client_model.parameters(), config.lr)
    batch_num = len(trainloaders[0])
    client_index= 0
    model_size = sum(p.numel() for p in client_model.parameters() if p.requires_grad)
    round_comm = 0
    total_comm = 0

    for round in range(config.global_round):
        logging.info(f"round: {round}")
        round_comm = 0

        for epoch in range(config.local_epoch):
            logging.info(f"epoch: {epoch}")
            
            for _ in range(batch_num):
                try:
                    one_batch_data = trainloaders_iter[client_index].next()
                except StopIteration:
                    trainloaders_iter[client_index] = iter(trainloaders[client_index])
                    one_batch_data = trainloaders_iter[client_index].next()
                inputs = one_batch_data[0].to(device)
                labels = one_batch_data[1].to(device)
                client_optim.zero_grad()
                server_optim.zero_grad()

                feature_map = client_model(inputs)
                round_comm += feature_map.numel()*2
                outputs = server_model(feature_map)
                loss = loss_fn(outputs, labels)
                loss.backward()
                client_optim.step()
                server_optim.step()
                client_index = (client_index+1)%config.num_worker
                round_comm += model_size

        total_comm += round_comm

        acc = eval_splited_model(client_model, server_model, testloader)
        logging.info(f"acc: {acc} | round_comm: {round_comm} | total_comm: {total_comm}")
        wandb.log({
            "round": round,
            "acc": acc,
            "round_comm": round_comm,
            "total_comm": total_comm
        })

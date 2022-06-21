
import sys
sys.path.append("..")
import torch
import logging
import wandb

from utils.data_utils import DatasetManager
from utils.model_utils import aggregate_model, eval_model, construct_model

def train(config, device):

    logging.info(f"@ vanilla_fl [{device}]")
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
    model_size = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    round_comm = 0
    total_comm = 0

    for round in range(config.global_round):
        logging.info(f"round: {round}")
        round_comm = 0
        [model.load_state_dict(global_model.state_dict()) for model in local_models]
        round_comm += model_size * len(local_models)

        for i, (model, trainloader, optim) in enumerate(zip(local_models, trainloaders, optims)):
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

        global_model = aggregate_model(local_models, [1/len(local_models)]*len(local_models))
        round_comm += model_size * len(local_models)
        total_comm += round_comm

        acc = eval_model(global_model, testloader)
        logging.info(f"acc: {acc} | round_comm: {round_comm} | total_comm: {total_comm}")
        wandb.log({
            "round": round,
            "acc": acc,
            "round_comm": round_comm,
            "total_comm": total_comm
        })

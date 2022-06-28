
import sys
sys.path.append("..")
import torch
import logging
import wandb

from utils.data_utils import DatasetManager
from utils.model_utils import aggregate_model, eval_model, construct_model, get_optim_params

def train(config, device):

    logging.info(f"@ ringsfl_v2 [{device}]")
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
    model_size = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    round_comm = 0
    total_comm = 0

    for round in range(config.global_round):
        logging.info(f"round: {round}")
        round_comm = 0
        [model.load_state_dict(global_model.state_dict()) for model in local_models]
        round_comm += model_size * len(local_models)

        for epoch in range(config.local_epoch):
            logging.info(f"epoch: {epoch}")
            for one_batch_datas in zip(*trainloaders):
                [optim.zero_grad() for optim in optims]

                for i, one_batch_data in enumerate(one_batch_datas):
                    inputs = one_batch_data[0].to(device)
                    labels = one_batch_data[1].to(device)

                    start = 0
                    for j in range(i, i + len(local_models)):
                        index = j % len(local_models)
                        inputs = local_models[index](inputs, start=start, stop=start+prop_lens[index])
                        start += prop_lens[index]
                        round_comm += inputs.numel()*2
                    
                    loss = loss_fn(inputs, labels)
                    loss.backward()
                
                [optim.step() for optim in optims]

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


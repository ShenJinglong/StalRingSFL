import torch
import logging
import wandb

from utils.data_utils import DatasetManager
from utils.model_utils import aggregate_model, eval_model, construct_model, get_optim_params
from utils.hardware_utils import get_free_gpu

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
wandb.init(
    project="StalRingSFL",
    entity="sjinglong"
)
config = wandb.config

DEVICE = f"cuda:{get_free_gpu()}" if torch.cuda.is_available() else "cpu"
prop_lens = [int(prop_len) for prop_len in config.prop_lens.split(":")]

dataset_manager = DatasetManager(config.dataset_name, "./datasets", config.block_num, config.batch_size)
if config.dataset_type == "iid":
    trainloaders = dataset_manager.get_iid_loaders(config.num_worker)
elif config.dataset_type == "noniid":
    trainloaders = dataset_manager.get_noniid_loaders(config.num_worker)
else:
    raise ValueError(f"Unrecognized dataset type: `{config.dataset_type}`")
testloader = dataset_manager.get_test_loader()

global_model = construct_model(config.model_type).to(DEVICE)
local_models = [construct_model(config.model_type).to(DEVICE) for _ in range(len(trainloaders))]
loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)
if config.alg == "ringsfl":
    optims = [torch.optim.SGD(optim_params, config.lr) for optim_params in get_optim_params(local_models, prop_lens)]
elif config.alg == "fl":
    optims = [torch.optim.SGD(model.parameters(), config.lr) for model in local_models]
else:
    raise ValueError(f"Unrecognized alg: `{config.alg}`")

if __name__ == "__main__":
    for round in range(config.global_round):
        logging.info(f"round: {round}")

        [model.load_state_dict(global_model.state_dict()) for model in local_models]

        for epoch in range(config.local_epoch):
            logging.info(f"epoch: {epoch}")
            for one_batch_datas in zip(*trainloaders):
                [optim.zero_grad() for optim in optims]

                if config.alg == "ringsfl":
                    for i, one_batch_data in enumerate(one_batch_datas):
                        inputs = one_batch_data[0].to(DEVICE)
                        labels = one_batch_data[1].to(DEVICE)

                        start = 0
                        for j in range(i, i + len(local_models)):
                            index = j % len(local_models)
                            inputs = local_models[index](inputs, start=start, stop=start+prop_lens[index])
                            start += prop_lens[index]
                        
                        loss = loss_fn(inputs, labels)
                        loss.backward()
                elif config.alg == "fl":
                    for model, one_batch_data in zip(local_models, one_batch_datas):
                        inputs = one_batch_data[0].to(DEVICE)
                        labels = one_batch_data[1].to(DEVICE)
                        outputs = model(inputs)
                        loss = loss_fn(outputs, labels)
                        loss.backward()
                
                [optim.step() for optim in optims]

        global_model = aggregate_model(local_models, [1/len(local_models)]*len(local_models))
        acc = eval_model(global_model, testloader)
        logging.info(f"acc: {acc}")
        wandb.log({
            "round": round,
            "acc": acc
        })

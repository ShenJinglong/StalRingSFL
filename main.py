
import logging

import torch

import wandb
from alg import (ringsfl_drop, ringsfl_v1, ringsfl_v2, splitfed, vanilla_fl,
                 vanilla_fl_drop, vanilla_sl, ringsfl_subring)
from utils.hardware_utils import get_free_gpu

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
wandb.init(
    project="StalRingSFL_Drop",
    entity="sjinglong"
)
config = wandb.config
DEVICE = f"cuda:{get_free_gpu()}" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    if config.alg == "ringsfl_v1":
        ringsfl_v1.train(config, DEVICE)
    elif config.alg == "ringsfl_v2":
        ringsfl_v2.train(config, DEVICE)
    elif config.alg == "vanilla_fl":
        vanilla_fl.train(config, DEVICE)
    elif config.alg == "splitfed":
        splitfed.train(config, DEVICE)
    elif config.alg == "vanilla_sl":
        vanilla_sl.train(config, DEVICE)
    elif config.alg == "ringsfl_drop":
        ringsfl_drop.train(config, DEVICE)
    elif config.alg == "vanilla_fl_drop":
        vanilla_fl_drop.train(config, DEVICE)
    elif config.alg == "ringsfl_subring":
        ringsfl_subring.train(config, DEVICE)
    else:
        raise ValueError(f"Unrecognized alg: `{config.alg}`")

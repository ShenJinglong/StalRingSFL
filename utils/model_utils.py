import sys
sys.path.append("..")
import collections
from typing import List
import torch
import copy

from model.MLP import *
from model.VGG16 import *
from model.ResNet18 import *
from model.CNN import *
from model.MobileNet import *

def aggregate_model(models: List[torch.nn.Module], weights: List[float]) -> torch.nn.Module:
    tmp_model = copy.deepcopy(models[0])
    global_dict = collections.OrderedDict()
    param_keys = tmp_model.state_dict().keys()
    for key in param_keys:
        sum = 0
        for weight, model in zip(weights, models):
            sum += weight * model.state_dict()[key]
        global_dict[key] = sum
    tmp_model.load_state_dict(global_dict)
    return tmp_model

def eval_model(model: torch.nn.Module, testloader: torch.utils.data.DataLoader) -> float:
    model.eval()
    device = next(model.parameters()).device
    total, correct = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # print(torch.nn.CrossEntropyLoss()(outputs, labels))
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    model.train()
    return 100 * correct / total

def get_optim_params(models: List[torch.nn.Module], prop_lens: List[int]) -> List[List]:
    optim_params = []
    for i in range(len(models)):
        params, start = [], 0
        for j in range(i, i + len(prop_lens)):
            index = j % len(prop_lens)
            params.extend(models[index].get_params(start=start, stop=start + prop_lens[index]))
            # print(f"([{index}] {start} {start + prop_lens[index]})", end=" ")
            start += prop_lens[index]
        # print()
        optim_params.append(params)
    return optim_params

def construct_model(
    model_type: str,
) -> torch.nn.Module:
    if model_type == "mlp":
        return MLP_Mnist()
    elif model_type == "vgg16":
        return VGG16_Cifar()
    elif model_type == "resnet18":
        return ResNet18_Cifar()
    elif model_type == "cnn":
        return CNN_Mnist()
    elif model_type == "mobilenet":
        return MobileNet_Mnist()
    elif model_type == "mobilenet_simple":
        return MobileNetSimple_Mnist()
    else:
        raise ValueError(f"Unrecognized model type: `{model_type}`")

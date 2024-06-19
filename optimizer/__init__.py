import copy
from typing import Any, Dict

import torch
import torch.nn as nn
from only_train_once import OTO

__all__ = ["build_optimizer"]


def build_optimizer(
    optim_config: Dict[str, Any],
    lr_scheduler_config: Dict[str, Any],
    epochs: int,
    step_each_epoch: int,
    model: nn.Module | OTO,
):
    from . import lr

    optim_config = copy.deepcopy(optim_config)
    if isinstance(model, OTO):
        optim = model.hesso(variant=optim_config.pop("name").lower(), **optim_config)
    else:
        optim = getattr(torch.optim, optim_config.pop("name"))(params=model.parameters(), **optim_config)

    lr_config = copy.deepcopy(lr_scheduler_config)
    lr_config.update({"epochs": epochs, "step_each_epoch": step_each_epoch})
    lr_scheduler = getattr(lr, lr_config.pop("name"))(**lr_config)(optimizer=optim)
    return optim, lr_scheduler

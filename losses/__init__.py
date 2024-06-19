import copy
from typing import Any, Dict

import torch.nn as nn

# det loss
from .det_db_loss import DBLoss

# rec loss
from .rec_ctc_loss import CTCLoss


def build_loss(conf: Dict[str, Any]) -> nn.Module:
    support_dict = ["DBLoss", "CTLoss"]

    conf = copy.deepcopy(conf)
    module_name = conf.pop("name")
    assert module_name in support_dict, Exception("loss only support {}".format(support_dict))
    module_class = eval(module_name)(**conf)
    return module_class

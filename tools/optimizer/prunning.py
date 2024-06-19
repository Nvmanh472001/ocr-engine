import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))


import torch
from only_train_once import OTO

from modeling.architectures import build_model
from tools.utility import Config, ConfigArguments
from utils.logging import get_logger

OUT_DIR = "./cache"


def main(cfg):
    logger = get_logger()
    global_config = cfg["Global"]
    model = build_model(cfg["Architecture"])
    print(global_config)

    if not os.path.exists(global_config["checkpoints"]):
        raise ValueError(f"Checkpoints not found at {global_config['checkpoints']}")

    model.load_state_dict(torch.load(global_config["checkpoints"]))

    dummy_input = torch.randn(1, 3, 960, 960)

    oto = OTO(model=model, dummy_input=dummy_input, strict_out_nodes=True)
    # oto.visualize(view=True, out_dir=OUT_DIR, display_params=False)

    full_flops = oto.compute_flops(in_million=True)["total"]
    full_num_params = oto.compute_num_params(in_million=True)

    oto.random_set_zero_groups()
    oto.mark_unprunable_by_node_ids(["node-847"])
    oto.construct_subnet(out_dir=OUT_DIR, ckpt_format="onnx")

    full_model_size = os.stat(oto.full_group_sparse_model_path)
    compressed_model_size = os.stat(oto.compressed_model_path)
    print("Size of full model        : ", full_model_size.st_size / (1024**3), "GBs")
    print("Size of compress model    : ", compressed_model_size.st_size / (1024**3), "GBs")

    # Compute FLOP and param for pruned model after oto.construct_subnet()
    pruned_flops = oto.compute_flops(in_million=True)["total"]
    pruned_num_params = oto.compute_num_params(in_million=True)

    print("FLOP  reduction (%): ", 1.0 - pruned_flops / full_flops)
    print("Param reduction (%): ", 1.0 - pruned_num_params / full_num_params)


if __name__ == "__main__":
    FLAGS = ConfigArguments().parse_args()
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop("opt")
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)
    main(cfg.cfg)

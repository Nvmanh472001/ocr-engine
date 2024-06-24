import argparse

from vietocr.model.trainer import Trainer
from vietocr.tool.config import Cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="./config/vgg-transformer-slim.yml")

    args = parser.parse_args()

    config = Cfg.load_config_from_file(args.config_path)
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()

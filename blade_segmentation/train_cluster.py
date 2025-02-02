import numpy as np
from src.eval_oneshot import train_clusterer
from omegaconf import DictConfig
from argparse import Namespace
from dotenv import load_dotenv
import hydra


@hydra.main(version_base=None, config_path="config_dir", config_name="config")
def main(cfg: DictConfig):
    args = Namespace(**cfg)
    train_clusterer(args)
    return 0

if __name__ == '__main__':
    load_dotenv(dotenv_path='../.env')
    main()
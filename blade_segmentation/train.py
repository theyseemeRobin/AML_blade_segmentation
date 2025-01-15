from src.train_RGB_cluster import train_rgb_cluster, train_rgb_cluster_parse_args
from src.file_handler import download_data, create_train_val_set
from omegaconf import DictConfig
from argparse import Namespace
from dotenv import load_dotenv
import hydra

@hydra.main(version_base=None, config_path="config_dir", config_name="config")
def main(cfg: DictConfig):
    download_data()
    create_train_val_set()
    args = Namespace(**cfg)
    train_rgb_cluster(args)


if __name__ == '__main__':
    load_dotenv(dotenv_path='../.env')
    main()
from src.train_RGB_cluster import train_rgb_cluster, train_rgb_cluster_parse_args
from src.file_handler import download_data
from dotenv import load_dotenv


def main(args):
    download_data()
    train_rgb_cluster(args)


if __name__ == '__main__':
    load_dotenv(dotenv_path='../.env')
    args = train_rgb_cluster_parse_args()
    main(args)
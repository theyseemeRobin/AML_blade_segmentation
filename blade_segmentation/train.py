from src.file_handler import download_data
from dotenv import load_dotenv
import os

def main():
    download_data()

if __name__ == '__main__':
    load_dotenv(dotenv_path='../.env')
    main()
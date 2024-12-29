import os
import zipfile
import gdown
import shutil

def download_data(data_path='data'):
    
    images_path = os.path.join(data_path, 'Images')
    videos_path = os.path.join(data_path, 'Videos')
    raw_data_path = os.path.join(data_path, 'raw_data.zip')

    # If the Images and Videos folders already exist, return
    if os.path.exists(images_path) and os.path.exists(videos_path):
        print('Data already downloaded.')
        return

    # Check if data_path exists
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        
    dataset_id = os.getenv('DATASET_ID')

    # Check if raw_data.zip exists
    if not os.path.exists(raw_data_path):
        gdown.download(id=dataset_id, output=raw_data_path)
        
    # Unzip the data
    with zipfile.ZipFile(raw_data_path, 'r') as zip_ref:
        zip_ref.extractall(data_path)
        
    # Get the name of the extracted folder
    extracted_folder = zip_ref.namelist()[0]
        
    # Unzip the inner zip files (images and videos) in the extracted folder into data_path
    for file in os.listdir(os.path.join(data_path, extracted_folder)):
        if file.endswith('.zip'):
            with zipfile.ZipFile(os.path.join(data_path, extracted_folder, file), 'r') as zip_ref:
                zip_ref.extractall(data_path)
                
    # Clean up the extracted folder
    shutil.rmtree(os.path.join(data_path, extracted_folder))

    # Clean up the raw_data.zip
    os.remove(raw_data_path)
    
    print('Data downloaded and unzipped.')

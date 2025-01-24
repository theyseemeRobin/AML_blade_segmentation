import os
import zipfile
import gdown
import shutil

def download_data(data_path='data/optic_thermal'):
    
    raw_data_path = os.path.join(data_path, 'raw_data.zip')

    # If the Images and Videos folders already exist, return
    if os.path.exists(data_path):
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
    

def create_train_val_set(data_path='data/optic_thermal'):
    images_path = os.path.join(data_path, 'Images')
    
    # Define source and target directories
    source_dirs = {
        'test-dev': ['Optical', 'Thermal'],
        'trainval': ['Optical', 'Thermal']
    }
    
    # If folders already exist, return
    if os.path.exists(os.path.join(images_path, 'test-dev', 'train')):
        print('Train and Val folders already exist.')
        return
    
    for main_folder in ['trainval']:
        for modality in ['Optical', 'Thermal']:
            
            # Source paths
            img_source = os.path.join(images_path, main_folder, modality)
            
            # Get sequences
            sequences = [d for d in os.listdir(img_source) if os.path.isdir(os.path.join(img_source, d))]
            
            # Calculate split
            split_idx = int(len(sequences) * 0.9)
            train_sequences = sequences[:split_idx]
            val_sequences = sequences[split_idx:]
            
            # Create and copy to train directories
            for seq in train_sequences:
                # Images
                train_img_dest = os.path.join(images_path, main_folder, 'train', modality, seq)
                os.makedirs(train_img_dest, exist_ok=True)
                shutil.copytree(os.path.join(img_source, seq), train_img_dest, dirs_exist_ok=True)
            
            # Create and copy to val directories
            for seq in val_sequences:
                # Images
                val_img_dest = os.path.join(images_path, main_folder, 'val', modality, seq)
                os.makedirs(val_img_dest, exist_ok=True)
                shutil.copytree(os.path.join(img_source, seq), val_img_dest, dirs_exist_ok=True)
        
            # Clean up the source directories   
            shutil.rmtree(os.path.join(images_path, main_folder, modality))
        
    print('Train and Val folders created.')
    # Display the number of sequences in the train and val folders
    for main_folder in ['test-dev', 'trainval']:
        for modality in ['Optical', 'Thermal']:
            train_path = os.path.join(images_path, main_folder, 'train', modality)
            val_path = os.path.join(images_path, main_folder, 'val', modality)
            print(f'{main_folder} - {modality} - Train: {len(os.listdir(train_path))}, Val: {len(os.listdir(val_path))}')
                
    
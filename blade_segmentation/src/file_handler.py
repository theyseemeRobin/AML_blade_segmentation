import os
import zipfile
import gdown
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

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
    if os.path.exists(os.path.join(images_path, 'trainval', 'train')):
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
            
def download_masks(data_path='data/optic_thermal'):
    
    masks_path = os.path.join(data_path, 'Images/test-dev/Masks_Optical')
    
    # If the Masks folder already exists, return
    if os.path.exists(masks_path):
        print('Masks already downloaded.')
        return
    
    dataset_id = os.getenv('DATASET_MASK_ID')

    # Check if masks.zip exists
    if not os.path.exists(masks_path):
        gdown.download(id=dataset_id, output=masks_path)
        
    # Unzip the masks in data_path directly
    with zipfile.ZipFile(masks_path, 'r') as zip_ref:
        zip_ref.extractall('data/')
        
    # Clean up the masks.zip
    os.remove(masks_path)
    
    print('Masks downloaded and unzipped.')
                
def convert_masks_to_davis_format(base_path='data/optic_thermal'):
    # Define mask locations to process
    mask_locations = [
        ('test-dev', 'Images/test-dev/Masks_Optical'),
        ('trainval-val', 'Images/trainval/val/Masks_Optical')
    ]
    
    # Create DAVIS palette (first color=background, second=object)
    davis_palette = np.zeros((256, 3), dtype=np.uint8)
    davis_palette[1] = [255, 0, 0]  # Red for object (index 1)

    for set_name, rel_path in mask_locations:
        masks_optical_path = os.path.join(base_path, rel_path)
        davis_masks_path = os.path.join(base_path, rel_path.replace('Masks_Optical', 'DAVIS_Masks'))
        
        if not os.path.exists(masks_optical_path):
            print(f"Skipping {set_name} - masks not found at {masks_optical_path}")
            continue
            
        if os.path.exists(davis_masks_path):
            print(f"DAVIS masks already exist for {set_name}")
            continue
            
        print(f"Processing {set_name} masks...")
        
        # Create directory structure mirror
        for root, dirs, files in os.walk(masks_optical_path):
            # Create corresponding DAVIS_Masks directory
            relative_path = os.path.relpath(root, masks_optical_path)
            davis_dir = os.path.join(davis_masks_path, relative_path)
            os.makedirs(davis_dir, exist_ok=True)

            # Process mask files
            for f in tqdm(files, desc=f'Processing {relative_path}'):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    src_path = os.path.join(root, f)
                    dst_path = os.path.join(davis_dir, f)
                    
                    try:
                        # Load RGBA mask
                        with Image.open(src_path) as img:
                            rgba = np.array(img.convert('RGBA'))
                            
                        # Create binary mask from alpha channel
                        alpha = rgba[..., 3]
                        mask = np.where(alpha > 0, 1, 0).astype(np.uint8)
                        
                        # Create indexed image
                        davis_img = Image.fromarray(mask, mode='P')
                        davis_img.putpalette(davis_palette.flatten())
                        
                        # Save with same filename
                        davis_img.save(dst_path)
                        
                    except Exception as e:
                        print(f"Error processing {src_path}: {str(e)}")
                        
        print(f"Completed processing {set_name}. Masks saved to {davis_masks_path}\n")
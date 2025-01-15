import os
from datetime import datetime
from src.data import Dataloader


def setup_dataset(args):
    resolution = args.resolution  # h,w
    if args.dataset == 'DAVIS':
        basepath = args.basepath
        img_dir = basepath + '/JPEGImages/480p'
        gt_dir = basepath + '/Annotations/480p'
        val_seq = [
            'dog', 'cows', 'goat', 'camel', 'libby', 'parkour', 'soapbox', 'blackswan', 'bmx-trees',
            'kite-surf', 'car-shadow', 'breakdance', 'dance-twirl', 'scooter-black', 'drift-chicane',
            'motocross-jump', 'horsejump-high', 'drift-straight', 'car-roundabout', 'paragliding-launch'
        ]
        val_data_dir = [img_dir, img_dir, gt_dir]

    elif args.dataset == 'DAVIS2017':
        basepath = args.basepath
        img_dir = basepath + '/JPEGImages/480p'
        gt_dir = basepath + '/Annotations/480p'
        val_seq = [
            'dog', 'cows', 'goat', 'camel', 'libby', 'parkour', 'soapbox', 'blackswan', 'bmx-trees',
            'kite-surf', 'car-shadow', 'breakdance', 'dance-twirl', 'scooter-black', 'drift-chicane',
            'motocross-jump', 'horsejump-high', 'drift-straight', 'car-roundabout', 'paragliding-launch',
            'bike-packing', 'dogs-jump', 'gold-fish', 'india', 'judo', 'lab-coat', 'loading', 'mbike-trick',
            'pigs', 'shooting'
        ]
        val_data_dir = [img_dir, img_dir, gt_dir]

    elif args.dataset == 'FBMS':
        basepath = args.basepath
        img_dir = args.basepath + '/FBMS/'
        gt_dir = args.basepath + '/Annotations/'
        val_seq = [
            'camel01', 'cars1', 'cars10', 'cars4', 'cars5', 'cats01', 'cats03', 'cats06',
            'dogs01', 'dogs02', 'farm01', 'giraffes01', 'goats01', 'horses02', 'horses04',
            'horses05', 'lion01', 'marple12', 'marple2', 'marple4', 'marple6', 'marple7', 'marple9',
            'people03', 'people1', 'people2', 'rabbits02', 'rabbits03', 'rabbits04', 'tennis'
        ]
        val_img_dir = args.basepath + '/FBMS/'
        val_gt_dir =args.basepath + '/FBMS_annotation/'
        val_data_dir = [val_img_dir, val_img_dir, val_gt_dir]

    elif args.dataset == 'STv2':
        basepath = args.basepath
        img_dir = basepath + '/STv2_img/JPEGImages/'
        gt_dir = basepath + '/STv2_gt&pred/STv2_gt/GroundTruth/'

        val_seq = [
            'drift', 'birdfall', 'girl', 'cheetah', 'worm', 'parachute', 'monkeydog',
            'hummingbird', 'soldier', 'bmx', 'frog', 'penguin', 'monkey', 'bird_of_paradise'
        ]
        val_data_dir = [img_dir, img_dir, gt_dir]

    elif args.dataset == 'YTVIS':
        basepath = args.basepath
        img_dir = basepath + '/val/JPEGImages'
        gt_dir = basepath + '/val/Annotations'
        val_seq = os.listdir(img_dir)
        val_data_dir = [img_dir, img_dir, gt_dir]

    elif args.dataset == 'turbines_O':
        basepath = args.basepath
        img_dir = basepath + '/JPEGImages'
        gt_dir = basepath + '/Annotations'
        val_seq = os.listdir(img_dir)
        val_data_dir = [img_dir, img_dir, gt_dir]

    elif args.dataset == 'turbines_OT':
        basepath = args.basepath
        img_dir = basepath + '/train/Optical'
        gt_dir = basepath + '/train/Masks_Optical'
        
        val_gt_dir = basepath + '/val/Masks_Optical'                      # NOTE (Robin): Annotations do not exist at the moment
        val_img_dir = basepath + '/val/Optical'
        val_seq = os.listdir(val_img_dir)
        
        val_data_dir = [val_img_dir, val_img_dir, gt_dir]
    else:
        raise ValueError('Unknown Setting.')
    
    flow_dir = basepath
    data_dir = [flow_dir, img_dir, gt_dir]
    trn_dataset = Dataloader(
        data_dir=data_dir,
        dataset=args.dataset,
        resolution=resolution,
        gap=args.gap,
        seq_length=args.num_frames,
        train=True
    )
    val_dataset = Dataloader(
        data_dir=val_data_dir,
        dataset=args.dataset,
        resolution=resolution,
        gap=args.gap,
        train=False,
        seq_length=args.num_frames,
        val_seq=val_seq
    )
    in_out_channels = 3
    
    return [trn_dataset, val_dataset, resolution, in_out_channels]


# NOTE (Robin): This function was  called in train_RGB_cluster.py, but did not exist. I'm not sure what it is supposed
# to do, but based on the name, how it is called, and how the returned paths are used, this is my best guess.
def setup_path(args):

    run_id = args.__dict__.get('run_id', str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

    base_output_path = os.path.join(args.output_path, run_id)
    if not os.path.exists(base_output_path):
        os.makedirs(base_output_path)

    model_output_path = os.path.join(base_output_path, "trained_model")
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)

    tensorboard_log_path = os.path.join(base_output_path, "tensorboard_logs")
    if not os.path.exists(tensorboard_log_path):
        os.makedirs(tensorboard_log_path)

    results_output_path = os.path.join(base_output_path, "results")
    if not os.path.exists(results_output_path):
        os.makedirs(results_output_path)

    return tensorboard_log_path, model_output_path, results_output_path
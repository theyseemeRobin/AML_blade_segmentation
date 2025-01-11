# Blade segementation project for Advanced Machine Learning

## Usage

Install the required packages

```bash
pip install -r requirements.txt
```

Setup .env file with the following content

```bash
# .env
DATASET_ID="GOOGLE_DRIVE_FILE_ID"
# More to come soon
```

This code is based on [SSL-UVOS](https://github.com/shvdiwnkozbw/SSL-UVOS/tree/main). Download the pretrained model 
[here](https://drive.google.com/file/d/1UhSPueJGpV4di9SVlZDmz0KWkuigQApA/view)

Run the following script from the [blade_segmentation](blade_segmentation) directory:

```bash
python train.py --basepath data/optic_thermal/Images/trainval \
--batch_size 1 \
--grad_iter 0 \
--lr 1e-5 \
--output_path results \
--dino_path models/dino_deitsmall8_pretrain_full_checkpoint.pth \
--dataset turbines_OT
```

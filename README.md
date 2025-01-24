# Blade segementation project for Advanced Machine Learning

## Usage

Install the required packages

```bash
pip install -r requirements.txt
```

Setup .env file with the following content in the root directory:

```bash
# .env
DATASET_ID="GOOGLE_DRIVE_FILE_ID"
DATASET_MASK_ID="GOOGLE_DRIVE_FILE_ID"
WANDB_API_KEY="Your WANDB API key"
WANDB_ENTITY=aml-blade-segmentation
```

This code is based on [SSL-UVOS](https://github.com/shvdiwnkozbw/SSL-UVOS/tree/main). Download the pretrained model 
[here](https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_full_checkpoint.pth)

Run the following script from the [blade_segmentation](blade_segmentation) directory:

```bash
python train.py --config-name config.yaml
```

Update the desired config file as needed. For hyperparameter tuning, hydra's sweeper can be used. To use the sweeper
configuration in a config file for hyperparameter tuning, the train script should be executed with a `--multirun` flag:

```bash
python train.py --config-name config.yaml --multirun
```

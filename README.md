# Text-to-Audio Generation

## Content of this repo

1. Training: The training of AudioLDM model, also with VAE model training code.
2. Evaluation: The code for evaluation on the AudioCaps dataset. Result saved as json files. Evaluation metrics including FAD, KL, IS, etc.
3. Preprocessed Audiocaps dataset and checkpoints.

## Prepare running environment
```shell 
# Create conda environment
conda create -n audioldm_train python=3.10
conda activate audioldm_train
# Clone the repo
git clone https://github.com/haoheliu/AudioLDM-training-finetuning.git; cd AudioLDM-training-finetuning
# Install running environment
pip install poetry
poetry install
```

## Download checkpoints and dataset
1. Download checkpoints


## Train the AudioLDM model
```python
# Train the AudioLDM (latent diffusion part)
python3 audioldm_train/train/latent_diffusion.py -c audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_crossattn_flant5.yaml

# Train the VAE
python3 audioldm_train/train/autoencoder.py -c audioldm_train/config/2023_11_13_vae_autoencoder/16k_64.yaml
```

The program will perform generation on the evaluation set every 15 epochs of training. After obtaining the audio generation folders (named val_<training-steps>), you can proceed to the next step for model evaluation.

## Evaluate the model output
Automatically evaluation based on each of the folder with generated audio
```python

# Evaluate all existing generated folder
python3 eval.py --log_path all

# Evaluate only a specific experiment folder
python3 eval.py --log_path <path-to-the-experiment-folder>
```
The evaluation result will be saved in a json file at the same level of the audio folder.

# Text-to-Audio Generation

## Content of this repo

1. Dataset: AudioCaps training, evaluation set, and their metadata. 
2. Pretrained model: Pretrained CLAP model, AudioMAE model, mel-spectrogram VAE model, HiFiGAN Vocoders.
3. Training: The training of AudioLDM model (LDM part).
4. Evaluation: The code for evaluation on the AudioCaps dataset. Result saved as json files.

## Prepare running environment
```shell 
# Create conda environment
conda create -n audioldm python=3.9
conda activate audioldm
# Install environment (if there are any dependencies missing, simply install them)
pip3 install audioldm
pip3 install taming-transformers-rom1504 kornia braceexpand webdataset wget ruamel.yaml
# Install dependencies for evaluation
git clone -b passt_replace_panns https://github.com/haoheliu/audioldm_eval.git; cd audioldm_eval; pip install -e .; pip install -e 'git+https://github.com/kkoutini/passt_hear21#egg=hear21passt';
```

## Download dataset and checkpoints


## Train the AudioLDM model
```python
# The original AudioLDM
python3 train_latent_diffusion.py -c config/2023_08_23_reproduce/audioldm_original.yaml
# A variant of AudioLDM that use FLAN-T5 text embedding as condition
python3 train_latent_diffusion.py -c config/2023_08_23_reproduce/audioldm_crossattn_flant5.yaml
# The log will be saved at ./log/latent_diffusion
```
The program will perform generation on the evaluation set every 15 epochs of training. After obtaining the audio generation folders (named val_<training-steps>), you can proceed to the next step for model evaluation.

## Evaluate the model output
```python
# Automatically evaluate all output result
python3 eval.py -l log/2023_08_23_reproduce
```
The evaluation result will be saved in a json file at the same level of the audio folder.

> Code organized by Haohe Liu. Do not distribute.
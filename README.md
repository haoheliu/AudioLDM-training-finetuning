# Train or Finetune the AudioLDM model

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
1. Download checkpoints from Google Drive: [link](https://drive.google.com/file/d/1pGw9T80YjU-Q-W4pd8c1hqJlauMIEdAm/view?usp=sharing).
2. Uncompress the checkpoint tar file and place the content into data/checkpoints/
3. Download the preprocessed AudioCaps from Google Drive: [link]()
4. Similarly, uncompress the dataset tar file and place the content into data/dataset

To double check if dataset or checkpoints are ready, run the following command:
```shell
python3 tests/validate_dataset_checkpoint.py
```
If the structure is not correct or partly missing. You will see the error message.

## Train the AudioLDM model
```python
# Train the AudioLDM (latent diffusion part)
python3 audioldm_train/train/latent_diffusion.py -c audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original.yaml

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

## Cite this work
If you found this tool useful, please consider citing

```bibtex
@article{liu2023audioldm,
  title={{AudioLDM}: Text-to-Audio Generation with Latent Diffusion Models},
  author={Liu, Haohe and Chen, Zehua and Yuan, Yi and Mei, Xinhao and Liu, Xubo and Mandic, Danilo and Wang, Wenwu and Plumbley, Mark D},
  journal={Proceedings of the International Conference on Machine Learning},
  year={2023}
}
```

```bibtex
@article{liu2023audioldm2,
  title={{AudioLDM 2}: Learning Holistic Audio Generation with Self-supervised Pretraining},
  author={Haohe Liu and Qiao Tian and Yi Yuan and Xubo Liu and Xinhao Mei and Qiuqiang Kong and Yuping Wang and Wenwu Wang and Yuxuan Wang and Mark D. Plumbley},
  journal={arXiv preprint arXiv:2308.05734},
  year={2023}
}
```


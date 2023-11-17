[![arXiv](https://img.shields.io/badge/arXiv-2301.12503-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2301.12503) [![arXiv](https://img.shields.io/badge/arXiv-2308.05734-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2308.05734)

# 🔊 AudioLDM training, finetuning, inference and evaluation

- [Prepare Python running environment](#prepare-python-running-environment)
  * [Download checkpoints and dataset](#download-checkpoints-and-dataset)
- [Play around with the code](#play-around-with-the-code)
  * [Train the AudioLDM model](#train-the-audioldm-model)
  * [Finetuning of the pretrained model](#finetuning-of-the-pretrained-model)
  * [Evaluate the model output](#evaluate-the-model-output)
  * [Inference with the pretrained model](#inference-with-the-pretrained-model)
  * [Train the model using your own dataset](#train-the-model-using-your-own-dataset)
- [Cite this work](#cite-this-work)

# Prepare Python running environment

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
1. Download checkpoints from Google Drive: [link](https://drive.google.com/file/d/1T6EnuAHIc8ioeZ9kB1OZ_WGgwXAVGOZS/view?usp=drive_link). The checkpoints including pretrained VAE, AudioMAE, CLAP, 16kHz HiFiGAN, and 48kHz HiFiGAN.
2. Uncompress the checkpoint tar file and place the content into **data/checkpoints/**
3. Download the preprocessed AudioCaps from Google Drive: [link](https://drive.google.com/file/d/16J1CVu7EZPD_22FxitZ0TpOd__FwzOmx/view?usp=drive_link)
4. Similarly, uncompress the dataset tar file and place the content into **data/dataset**

To double check if dataset or checkpoints are ready, run the following command:
```shell
python3 tests/validate_dataset_checkpoint.py
```
If the structure is not correct or partly missing. You will see the error message.

# Play around with the code

## Train the AudioLDM model
```python
# Train the AudioLDM (latent diffusion part)
python3 audioldm_train/train/latent_diffusion.py -c audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original.yaml

# Train the VAE (Optional)
# python3 audioldm_train/train/autoencoder.py -c audioldm_train/config/2023_11_13_vae_autoencoder/16k_64.yaml
```

The program will perform generation on the evaluation set every 5 epochs of training. After obtaining the audio generation folders (named val_<training-steps>), you can proceed to the next step for model evaluation.

## Finetuning of the pretrained model

You can finetune with two pretrained checkpoint, first download the one that you like (e.g., using wget):
1. Medium size AudioLDM: https://zenodo.org/records/7884686/files/audioldm-m-full.ckpt
2. Small size AudioLDM: https://zenodo.org/records/7884686/files/audioldm-s-full

Place the checkpoint in the *data/checkpoints* folder

Then perform finetuning with one of the following commands:
```shell
# Medium size AudioLDM
python3 audioldm_train/train/latent_diffusion.py -c audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml --reload_from_ckpt data/checkpoints/audioldm-m-full.ckpt

# Small size AudioLDM
python3 audioldm_train/train/latent_diffusion.py -c audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original.yaml --reload_from_ckpt data/checkpoints/audioldm-s-full
```
You can specify your own dataset following the same format as the provided AudioCaps dataset.

Note that the pretrained AudioLDM checkpoints are under CC-by-NC 4.0 license, which is not allowed for commerial use.

## Evaluate the model output
Automatically evaluation based on each of the folder with generated audio
```python

# Evaluate all existing generated folder
python3 audioldm_train/eval.py --log_path all

# Evaluate only a specific experiment folder
python3 audioldm_train/eval.py --log_path <path-to-the-experiment-folder>
```
The evaluation result will be saved in a json file at the same level of the audio folder.

## Inference with the pretrained model
Use the following syntax:

```shell
python3 audioldm_train/infer.py --config_yaml <The-path-to-the-same-config-file-you-use-for-training> --list_inference <the-filelist-you-want-to-generate>
```

For example:
```shell
# Please make sure you have train the model using audioldm_crossattn_flant5.yaml
# The generated audio will be saved at the same log folder if the pretrained model.
python3 audioldm_train/infer.py --config_yaml audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_crossattn_flant5.yaml --list_inference tests/captionlist/inference_test.lst
```

The generated audio will be named with the caption by default. If you like to specify the filename to use, please checkout the format of *tests/captionlist/inference_test_with_filename.lst*.

This repo only support inference with the model you trained by yourself. If you want to use the pretrained model directly, please use these two repos: [AudioLDM](https://github.com/haoheliu/AudioLDM) and [AudioLDM2](https://github.com/haoheliu/AudioLDM2).

## Train the model using your own dataset
Super easy, simply follow these steps:

1. Prepare the metadata with the same format as the provided AudioCaps dataset. 
2. Register in the metadata of your dataset in **data/dataset/metadata/dataset_root.json**
3. Use your dataset in the YAML file.

You do not need to resample or pre-segment the audiofile. The dataloader will do most of the jobs.

# Cite this work
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

# Acknowledgement
We greatly appreciate the open-soucing of the following code bases. Open source code base is the real-world infinite stone 💎!
- https://github.com/CompVis/stable-diffusion
- https://github.com/LAION-AI/CLAP
- https://github.com/jik876/hifi-gan

> This research was partly supported by the British Broadcasting Corporation Research and Development, Engineering and Physical Sciences Research Council (EPSRC) Grant EP/T019751/1 "AI for Sound", and a PhD scholarship from the Centre for Vision, Speech and Signal Processing (CVSSP), Faculty of Engineering and Physical Science (FEPS), University of Surrey. For the purpose of open access, the authors have applied a Creative Commons Attribution (CC BY) license to any Author Accepted Manuscript version arising. We would like to thank Tang Li, Ke Chen, Yusong Wu, Zehua Chen and Jinhua Liang for their support and discussions.


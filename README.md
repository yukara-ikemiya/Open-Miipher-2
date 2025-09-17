# 🎵 Open-Miipher-2 | A Universal Speech Restoration Model

This is an unofficial implementation of **`Miipher-2`**[1]
which is **a state-of-the-art universal speech restoration model** from Google Research.

![Miipher-2](./assets/fig/[TBD])

This repository supports:
- 🔥 Full implementation and training code for the `Miipher-2` model
- 🔥 [TBD: Add supported features]
- 🔥 Distributed training with multiple GPUs / multiple Nodes

## What Google Can Do vs. What You Can Do

Google's Miipher-2 leverages proprietary datasets and large-scale infrastructure that are not publicly accessible. While you cannot use Google's internal data or resources, this repository enables you to train and experiment with open datasets and models. The implementation is designed to be flexible and reproducible for academic and general-purpose use.

## Google USM

Google uses the Universal Speech Model (USM) [4] for both feature extraction and as the base (pretrained) model for the feature cleaner in Miipher-2. Specifically, Miipher-2 is trained using the first 13 layers of a 32-layer Conformer with 2 billion parameters. However, this pretrained model is not publicly available.

On the other hand, Google has open-sourced [Gemma 3](https://huggingface.co/docs/transformers/main/model_doc/gemma3), a multimodal LLM that includes a 0.6 billion parameter (12-layer) USM module. In this repository, the default configuration uses up to the 6th layer of this model as the base for the feature cleaner. Naturally, differences in base model size may lead to variations in restoration performance between the Google version and this repository.

The key differences between the Google version and this repository are summarized below:

|            | Base model | Model size | Conformer layers | Dimension | Frame rate | Parallel Adapter size |
|:-----------|:----------:|:----------:|:----------------:|:---------:|:----------:|:---:|
| **Google** | USM        | 2B         | 13th / 32 layers | 1536      | 25         | 41M |
| **Open-Miipher-2** | USM     | 0.6B       | 6th / 12 layers  | 1536      | 25         | 19M |

For more details on the selection of Conformer layers and related considerations, please refer to the [Tips](#-tips) section.


## Audio dataset
According to the paper, Google used `3,195 hours of speech from 1,642 speakers across 44 languages` as speech data and `internally collected audio snippets from environments such as cafes, kitchens, and automobiles` as noise data for training Miipher-2. These datasets are internal to Google and are not publicly available.

For general-purpose use, it is preferable to utilize larger and more diverse speech/noise datasets. However, for experiments or academic purposes, you can use open datasets such as those listed below.

| Type  | Dataset name | Link | Hours |
|-------|-------------------------|------|---|
| Speech | LibriTTS-R [5] | [https://www.openslr.org/141/](https://www.openslr.org/141/) | 585 |
| Noise  | TAU Urban Acoustic Scenes 2020 Mobile, Development dataset | [https://zenodo.org/records/3670167](https://zenodo.org/records/3670167) | 64 |
| Noise  | TAU Urban Audio-Visual Scenes 2021, Development dataset | [https://zenodo.org/records/4477542](https://zenodo.org/records/4477542) | 34 |

# Requirements

- Python 3.8.10 or later
- PyTorch 2.1 or later
- transformers>=4.53 (NOTE: `transformers` must contain Gemma implementations from Google.)

## Building a training environment

To simplify setting up the training environment, I recommend to use container systems like `Docker` or `Singularity` instead of installing dependencies on each GPU machine. Below are the steps for creating `Singularity` containers. 

All example scripts are stored at the [container](container/) folder.

### 1. Install Singularity

Install the latest Singularity by following the official instruction.
- https://docs.sylabs.io/guides/main/user-guide/quick_start.html#quick-installation-steps

### 2. Create a Singularity image file

Create (build) a Singularity image file with a definition file.
```bash
singularity build --fakeroot Open-Miipher-2.sif Open-Miipher-2.def
```

** NOTE: You might need to change NVIDIA base image in the definition file to match your GPU machine.

Now, you obtained a container file for training and inference of Open-Miipher-2.

## Setting a WandB account for logging

The training code also requires a Weights & Biases account to log the training outputs and demos. 
Please create an account and follow the instruction.

Once you create your WandB account,
you can obtain the API key from https://wandb.ai/authorize after logging in to your account.
And then, the API key can be passed as an environment variable `WANDB_API_KEY` to a training job
for logging training information.

```bash
$ WANDB_API_KEY="12345x6789y..."
```



# Data preparation

resampling の計算を避けるために 24khz のオーディオを用意してロードすることをおすすめする。

本repository では 高速な dataset class の初期化のため、各オーディオデータセットの root directory に配置された metadata file (`metadata.csv`) を利用しており、これを作成するには以下の各ディレクトリに対して以下のスクリプトを実行すればよい。

```bash
AUDIO_DIR=/path/to/audio-root-directory/
python dataset/script/make_metadata_csv.py --root-dir ${AUDIO_DIR}
```

## Pre-computation of degraded speech signals [Not supported]

For clarity and simplicity, this repository applies random degradations to audio samples within the Dataset during loading (online processing). However, some degradation methods (e.g., convolution filtering) can be computationally intensive on the CPU, potentially becoming a bottleneck and preventing full utilization of GPU resources during training. In such cases, consider pre-computing and saving multiple degraded versions of each clean speech sample before training, so that the Dataset can load these pre-processed files directly.

# Training

Miipher-2 training consists of the following three stages:

1. Training of the feature cleaner module
2. Pretraining of the WaveFit module
3. Finetuning of the WaveFit module

Stage 1 trains a module that converts noisy SSL features into clean ones, serving as the main part of audio restoration. Stage 2 pre-trains the WaveFit speech vocoder using only clean speech. Stage 3 fine-tunes the WaveFit module using features restored by the feature cleaner from Stage 1.

For Stage 2, short audio signals of 0.6 seconds are used to accelerate training (see Sec.2.4 in [2]). In contrast, Stages 1 and 3 use longer audio signals (typically 10–30 seconds) to capture more global information. The original paper suggests that the maximum input length of the USM Conformer block is 30 seconds, but this requires significant memory and may be impractical for most users. Therefore, the sample code in this repository defaults to a 10-second input length.

In summary, the input signal lengths for each training stage are as follows:

| Stage | Module                | Input length | Input audio | Purpose                                      |
|-------|-----------------------|--------------|------|----------------------------------------------|
| 1     | Feature Cleaner       | 10 sec  | Noisy speech | Restore noisy SSL features |
| 2     | WaveFit Pretraining  | 0.6 sec | Clean speech | Pretrain vocoder with clean features |
| 3     | WaveFit Finetuning   | 10 sec  | Noisy speech | Finetune vocoder with restored features |


## Stage 1: Feature Cleaner Training

```bash
ROOT_DIR="/path/to/this/repository/"
CONTAINER_PATH="/path/to/Open-Miipher-2.sif"
DATASET_ROOT="/path/to/dataset/root/"
JOB_ID="your_job_id"

# Dataset
DIRS_AUDIO=${DATASET_ROOT}/LibriTTS_R/train-clean-100/
DIRS_NOISE=${DATASET_ROOT}/TAU-urban-audio-visual-scenes-2021-development_24k-mono/,${DATASET_ROOT}/TAU-urban-acoustic-scenes-2020-mobile-development_24k-mono/

# Configuration
MODEL="feature_cleaner/google-usm"
PROJECT_NAME="cleaner_google-usm"
DATA="deg_gemma_16khz_10sec"
OPTIMIZER="feature_cleaner"
BS_PER_GPU=20
NUM_WORKERS=4
EXTRA_ARGS="model=${MODEL} data=${DATA} optimizer=${OPTIMIZER}"

WANDB_API_KEY="your_wandb_api_key"
HF_TOKEN="your_huggingface_token"
PORT=50000

OUTPUT_DIR=${ROOT_DIR}/runs/train/${MODEL}/${JOB_ID}
mkdir -p ${OUTPUT_DIR}

# Calculate total batch size based on number of GPUs
NUM_GPUS=2  # Adjust based on your setup
BATCH_SIZE=$((${NUM_GPUS}*${BS_PER_GPU}))

singularity exec --nv --pwd $ROOT_DIR -B $ROOT_DIR -B $DATASET_ROOT \
    --env HYDRA_FULL_ERROR=1 --env MASTER_PORT=${PORT} \
    --env WANDB_API_KEY=$WANDB_API_KEY --env HF_TOKEN=$HF_TOKEN \
    ${CONTAINER_PATH} \
torchrun --nproc_per_node gpu --master_port ${PORT} \
${ROOT_DIR}/src/train.py \
    data.train.dirs_audio=[${DIRS_AUDIO}] \
    data.train.dirs_noise=[${DIRS_NOISE}] \
    trainer.output_dir=${OUTPUT_DIR} \
    trainer.batch_size=${BATCH_SIZE} \
    trainer.num_workers=${NUM_WORKERS} \
    trainer.logger.project_name=${PROJECT_NAME} \
    trainer.logger.run_name=job-${JOB_ID} \
    ${EXTRA_ARGS}
```

## Stage 2: WaveFit Pretraining

```bash
ROOT_DIR="/path/to/this/repository/"
CONTAINER_PATH="/path/to/Open-Miipher-2.sif"
DATASET_ROOT="/path/to/dataset/root/"
JOB_ID="your_job_id"

# Dataset
DIRS_AUDIO=${DATASET_ROOT}/LibriTTS_R/train-clean-100/

# Configuration
PROJECT_NAME="wavefit_pretrain"
MODEL="miipher-2_google-usm_wavefit-5_clean-input"
DATA="deg_gemma_24khz_06sec_clean-only"
OPTIMIZER="wavefit"
BS_PER_GPU=30
NUM_WORKERS=4
EXTRA_ARGS="model=${MODEL} data=${DATA} optimizer=${OPTIMIZER}"

WANDB_API_KEY="your_wandb_api_key"
HF_TOKEN="your_huggingface_token"
PORT=50000

OUTPUT_DIR=${ROOT_DIR}/runs/train/${MODEL}/${JOB_ID}
mkdir -p ${OUTPUT_DIR}

# Calculate total batch size based on number of GPUs
NUM_GPUS=2  # Adjust based on your setup
BATCH_SIZE=$((${NUM_GPUS}*${BS_PER_GPU}))

singularity exec --nv --pwd $ROOT_DIR -B $ROOT_DIR -B $DATASET_ROOT \
    --env HYDRA_FULL_ERROR=1 --env MASTER_PORT=${PORT} \
    --env WANDB_API_KEY=$WANDB_API_KEY --env HF_TOKEN=$HF_TOKEN \
    ${CONTAINER_PATH} \
torchrun --nproc_per_node gpu --master_port ${PORT} \
${ROOT_DIR}/src/train.py \
    data.train.dirs_audio=[${DIRS_AUDIO}] \
    trainer.output_dir=${OUTPUT_DIR} \
    trainer.batch_size=${BATCH_SIZE} \
    trainer.num_workers=${NUM_WORKERS} \
    trainer.logger.project_name=${PROJECT_NAME} \
    trainer.logger.run_name=job-${JOB_ID} \
    ${EXTRA_ARGS}
```

## Stage 3: WaveFit Finetuning

```bash
ROOT_DIR="/path/to/this/repository/"
CONTAINER_PATH="/path/to/Open-Miipher-2.sif"
DATASET_ROOT="/path/to/dataset/root/"
JOB_ID="your_job_id"

# Dataset
DIRS_AUDIO=${DATASET_ROOT}/LibriTTS_R/train-clean-100/
DIRS_NOISE=${DATASET_ROOT}/TAU-urban-audio-visual-scenes-2021-development_24k-mono/,${DATASET_ROOT}/TAU-urban-acoustic-scenes-2020-mobile-development_24k-mono/

# Configuration
PROJECT_NAME="wavefit_finetune"
MODEL="miipher-2_google-usm_wavefit-5_noisy-input"
DATA="deg_gemma_16khz_10sec"
OPTIMIZER="wavefit"
BS_PER_GPU=5
NUM_WORKERS=4

# Pre-trained model checkpoints
FEATURE_CLEANER_CKPT_DIR=${ROOT_DIR}/runs/_debug/train/feature_cleaner/google-usm/135294/ckpt/latest/
VOCODER_CKPT_DIR=${ROOT_DIR}/runs/_debug/train/miipher-2_google-usm_wavefit-5_clean-input/134297/ckpt/latest/

EXTRA_ARGS="model=${MODEL} data=${DATA} optimizer=${OPTIMIZER}"
EXTRA_ARGS="${EXTRA_ARGS} model.feature_cleaner_ckpt_dir=${FEATURE_CLEANER_CKPT_DIR} model.vocoder_ckpt_dir=${VOCODER_CKPT_DIR}"

WANDB_API_KEY="your_wandb_api_key"
HF_TOKEN="your_huggingface_token"
PORT=50000

OUTPUT_DIR=${ROOT_DIR}/runs/train/${MODEL}/${JOB_ID}
mkdir -p ${OUTPUT_DIR}

# Calculate total batch size based on number of GPUs
NUM_GPUS=2  # Adjust based on your setup
BATCH_SIZE=$((${NUM_GPUS}*${BS_PER_GPU}))

singularity exec --nv --pwd $ROOT_DIR -B $ROOT_DIR -B $DATASET_ROOT \
    --env HYDRA_FULL_ERROR=1 --env MASTER_PORT=${PORT} \
    --env WANDB_API_KEY=$WANDB_API_KEY --env HF_TOKEN=$HF_TOKEN \
    ${CONTAINER_PATH} \
torchrun --nproc_per_node gpu --master_port ${PORT} \
${ROOT_DIR}/src/train.py \
    data.train.dirs_audio=[${DIRS_AUDIO}] \
    data.train.dirs_noise=[${DIRS_NOISE}] \
    trainer.output_dir=${OUTPUT_DIR} \
    trainer.batch_size=${BATCH_SIZE} \
    trainer.num_workers=${NUM_WORKERS} \
    trainer.logger.project_name=${PROJECT_NAME} \
    trainer.logger.run_name=job-${JOB_ID} \
    trainer.debug=0 \
    ${EXTRA_ARGS}
```

## Resume training from a checkpoint

While training, checkpoints (state_dict) of models, optimizers and schedulers are saved under the output directory specified in the configuration as follows.
```
output_dir/
├─ ckpt/
│  ├─ latest/
│  │  ├─ model.pth
│  │  ├─ discriminator.pth
│  │  ├─ optimizer.pth
│  │  ├─ scheduler.pth
│  │  ├─ ...
```

By specifying the checkpoint directory, you can easily resume your training from the checkpoint.
```bash
CKPT_DIR="output_dir/ckpt/latest/"
OUTPUT_DIR="another/directory/"

# Execution
singularity exec --nv --pwd $ROOT_DIR -B $ROOT_DIR -B $DATASET_ROOT \
    --env MASTER_PORT=${PORT} --env WANDB_API_KEY=$WANDB_API_KEY \
    ${CONTAINER_PATH} \
torchrun --nproc_per_node gpu ${ROOT_DIR}/src/train.py \
    trainer.ckpt_dir=${CKPT_DIR} \
    trainer.output_dir=${OUTPUT_DIR}
```


<!-- # Inference

Using pre-trained Open-Miipher-2 models, you can perform inference with audio signals as input
(e.g. for speech restoration evaluation).

The [`inference.py`](src/inference.py) [TBD: check if this file exists] perform inference for all of audio files in a target directory.
To check other options for the script, please use `-h` option.

```bash
CKPT_DIR="output_dir/ckpt/latest/"
AUDIO_DIR="path/to/target/speech/directory/"

singularity exec --nv --pwd $ROOT_DIR -B $ROOT_DIR -B $AUDIO_DIR \
    --env MASTER_PORT=${PORT} \
    ${CONTAINER_PATH} \
torchrun --nproc_per_node gpu --master_port ${PORT} \
${ROOT_DIR}/src/inference.py \
    --ckpt-dir ${CKPT_DIR} \
    --input-audio-dir ${AUDIO_DIR} \
    --output-dir ${OUTPUT_DIR}
```` -->

# 💡 Tips

## A layer number of an audio encoder for extracting audio feature

[TBD: Description of optimal layer selection for audio feature extraction]

## Degradation types

The methods for degrading speech used for model training differ between those described in the paper and those implemented in this repository, as summarized below.

|                | Background noise | Room reverb | Codec (MP3, Vorbis, A-law, AMR-WB, OPUS) | Soft/Hard clipping | Lowpass |
|:--------------:|:---------------:|:-----------:|:----------------------------------------:|:------------------:|:-------:|
| **Google**     | ✓               | ✓           | ✓                                        |                    |         |
| **Open-Miipher-2** | ✓           | ✓           |                                          | ✓                  | ✓       |

Codec processing is considered too computationally intensive for online processing, so it is excluded from this repository.

## Frame-wise decoding of WaveFit in Miipher-2

TBD, 0.6 sec の学習 length、小さい音が学習時に含まれていない件について

# 🤔 Unclear points in the implementation

## 1. Parameter size

According to the paper, the Parallel Adapter (PA) is described as having `20 million` learnable parameters (Sec.2.2). However, the PA used in Miipher-2 takes a 1536-dimensional input and has a 1024-dimensional bottleneck structure, which is applied to 13 layers of Conformers. The approximate parameter size of the PA can be calculated from two linear layers, resulting in a total parameter count of about `1536 x 1024 x 2 x 13 = 40.9M`. This is likely a typo in the paper.

## 2. Upsampling method of USM feature

When inputting SSL features into WaveFit in Miipher-2, upsampling is performed along the time axis to fit the frame rate to the appropriate input length (Sec.2.3). The specific upsampling method (e.g., 'nearest', 'linear', or 'bilinear') is not described in the paper, but it is likely that the choice does not significantly affect performance. Therefore, this repository uses `linear` interpolation as the default.

## 3. Loss function of feature cleaner

The loss function for training feature cleaner is defined in the Miipher paper [2] as below.

```math
\mathcal{L} = \| S - \hat{S} \|_{1} 
+ \| S - \hat{S} \|_{2}^{2} 
+ \frac{\| S - \hat{S} \|_{2}^{2}}{\| S \|_{2}^{2}},
\quad \text{where  } 
\| S \|_{p} = \left( \sum_{k} \sum_{d} | S_{k,d} |^{p} \right)^{1/p}.
```

In the paper, the first term is referred to as "mean-absolute-error" and the second term as "mean-squared-error," but the formulas do not actually compute the mean. In practice, when calculating this loss, the first and second terms become disproportionately large compared to the third term (spectral convergence loss). Therefore, it is reasonable to compute the loss as follows:

```math
\mathcal{L} = \frac{1}{KD}\| S - \hat{S} \|_{1} 
+ \frac{1}{KD}\| S - \hat{S} \|_{2}^{2} 
+ \frac{\| S - \hat{S} \|_{2}^{2}}{\| S \|_{2}^{2}}
```

# References

1. "Miipher-2: A Universal Speech Restoration Model for Million-Hour Scale Data Restoration", S. Karita et al., WASPAA 2025
1. "Miipher: A Robust Speech Restoration Model Integrating Self-Supervised Speech and Text Representations", Y. Koizumi, WASPAA 2021
1. "WaveFit: An Iterative and Non-autoregressive Neural Vocoder based on Fixed-Point Iteration", Y. Koizumi et al., IEEE SLT, 2022
1. "Google USM: Scaling Automatic Speech Recognition Beyond 100 Languages", Y. Zhang et al., Arxiv, 2023
1. "LibriTTS-R: A Restored Multi-Speaker Text-to-Speech Corpus", Yuma Koizumi et al., INTERSPEECH 2023.
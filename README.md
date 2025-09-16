# 🎵 Open-Miipher-2 | A Universal Speech Restoration Model

This is an unofficial implementation of **`Miipher-2`**[1]
which is **a state-of-the-art universal speech restoration model** from Google Research.

![Miipher-2](./assets/fig/[TBD])

This repository supports:
- 🔥 Full implementation and training code for the `Miipher-2` model
- 🔥 [TBD: Add supported features]
- 🔥 Distributed training with multiple GPUs / multiple Nodes

# Requirements

- Python [TBD: version] or later
- PyTorch [TBD: version] or later

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

# What Google can do and You cannot do

## Google USM

[TBD: Description of Google USM capabilities and limitations]

## Audio dataset

speech dataset and noise dataset

For general purposes, you should use as many speech/noise audios as possible for training.

# Data preparation

resampling の計算を避けるために 24khz のオーディオを用意してロードすることをおすすめする。

本repository では 高速な dataset class の初期化のため、各オーディオデータセットの root directory に配置された metadata file (`metadata.csv`) を利用しており、これを作成するには以下の各ディレクトリに対して以下のスクリプトを実行すればよい。

```bash
AUDIO_DIR=/path/to/audio-root-directory/
python dataset/script/make_metadata_csv.py --root-dir ${AUDIO_DIR}
```

# Training

## Training from scratch

In this repository, all the training parameters are configured by `Hydra`,
allowing them to be set as command-line arguments.

The following is an example of a job script for training using the [TBD: dataset name] dataset.
```bash
ROOT_DIR="/path/to/this/repository/"
DATASET_DIR="/path/to/[TBD: dataset]/"
CONTAINER_PATH="/path/to/Open-Miipher-2.sif"
TRAIN_DIRS=${DATASET_DIR}/[TBD: train directories]
TEST_DIRS=${DATASET_DIR}/[TBD: test directories]

WANDB_API_KEY="12345x6789y..."
PORT=12345
JOB_ID="job_name"
OUTPUT_DIR=${ROOT_DIR}/output/${MODEL}/${JOB_ID}/

MODEL="[TBD: model name]"
BATCH_SIZE=[TBD: batch size]   # This must be a multiple of GPU number. Please adjust to your environment.
NUM_WORKERS=8

mkdir -p ${OUTPUT_DIR}

# Execution
singularity exec --nv --pwd $ROOT_DIR -B $ROOT_DIR -B $DATASET_DIR \
    --env MASTER_PORT=${PORT} --env WANDB_API_KEY=$WANDB_API_KEY \
    ${CONTAINER_PATH} \
torchrun --nproc_per_node gpu ${ROOT_DIR}/src/train.py \
    model=${MODEL} \
    data.train.dir_list=[${TRAIN_DIRS}] data.test.dir_list=[${TEST_DIRS}] \
    trainer.output_dir=${OUTPUT_DIR} \
    trainer.batch_size=${BATCH_SIZE} \
    trainer.num_workers=${NUM_WORKERS} \
    trainer.logger.project_name=${MODEL} \
    trainer.logger.run_name=job-${JOB_ID}
```
** Please note that the dataset directories are provided as lists.

## Resume training from a checkpoint

While training, checkpoints (state_dict) of models, optimizers and schedulers are saved under the output directory specified in the configuration as follows.
```
output_dir/
├─ ckpt/
│  ├─ latest/
│  │  ├─ [TBD: checkpoint files]
│  │  ├─ ...
```

By specifying the checkpoint directory, you can easily resume your training from the checkpoint.
```bash
CKPT_DIR="output_dir/ckpt/latest/"

# Execution
singularity exec --nv --pwd $ROOT_DIR -B $ROOT_DIR -B $DATASET_DIR \
    --env MASTER_PORT=${PORT} --env WANDB_API_KEY=$WANDB_API_KEY \
    ${CONTAINER_PATH} \
torchrun --nproc_per_node gpu ${ROOT_DIR}/src/train.py trainer.ckpt_dir=${CKPT_DIR}
```

### Overrides of parameters

When resuming training, you might want to override some configuration parameters.
To achieve this, in my implementation, only the specified parameters in job scripts will override the configuration from the checkpoint directory.

For example, in the following case, the checkpoint will be loaded from `CKPT_DIR`,
but the training outputs will be saved under `OUTPUT_DIR`.
```bash
CKPT_DIR="output_dir/ckpt/latest/"
OUTPUT_DIR="another/directory/"

# Execution
singularity exec --nv --pwd $ROOT_DIR -B $ROOT_DIR -B $DATASET_DIR \
    --env MASTER_PORT=${PORT} --env WANDB_API_KEY=$WANDB_API_KEY \
    ${CONTAINER_PATH} \
torchrun --nproc_per_node gpu ${ROOT_DIR}/src/train.py \
    trainer.ckpt_dir=${CKPT_DIR} \
    trainer.output_dir=${OUTPUT_DIR}
```

# Inference

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
````

# 💡 Tips

## A layer number of an audio encoder for extracting audio feature

[TBD: Description of optimal layer selection for audio feature extraction]

## Degradation types

[TBD: Description of different types of audio degradation that can be handled]

## Frame-wise decoding of WaveFit in Miipher-2

TBD, 0.6 sec の学習 length、小さい音が学習時に含まれていない件について

# 🤔 Unclear points in the implementation

## 1. Parameter size

[TBD: Discussion of model parameter sizes and their implications]

## 2. Upsampling method of USM feature

'nearest', 'linear'or 'bilinear' ?

## 3. Loss function of feature cleaner

The loss function for training feature cleaner is defined in the Miipher paper [2].
mean-absolute~ にするために要素数で割るの忘れている的な

## 4. [TBD: Additional unclear points]

[TBD: Add more unclear implementation details as needed]

# TODO

- [TBD: List of future improvements and features to be implemented]

# References

1. "Miipher-2: A Universal Speech Restoration Model for Million-Hour Scale Data Restoration", S. Karita et al., WASPAA 2025
2. "Miipher: A Robust Speech Restoration Model Integrating Self-Supervised Speech and Text Representations", Y. Koizumi, WASPAA 2021
3. "WaveFit: An Iterative and Non-autoregressive Neural Vocoder based on Fixed-Point Iteration", Y. Koizumi et al., IEEE SLT, 2022
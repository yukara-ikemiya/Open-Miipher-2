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
| **Google** | USM        | 2B         | 13th / 32 layers | 1536      | 25         | 40M |
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

Miipher-2 の学習は以下の3つのステージからなる。
1. Training of a feature cleaner module
2. Pre-training of a WaveFit module
3. Fine-tuning of a WaveFit module

`1.` は noisy な SSL feature を clean にする module の学習であり、audio restoration の main part である。
`2.` は speech vocoder である WaveFit module の pretraining であり、clean な speech のみで学習される。
`3.` は、 `1.` で学習された feature cleaner により restore された feature を入力として WaveFit module を finetuning する。

このとき、`2.` に関しては学習を高速化するために 0.6 [sec] という短いオーディオ信号を入力しているのに対し (Sec.2.4 in [2])、`1.`と`3.` については音声信号の大域的な情報も考慮して学習を行うために 10~30 [sec] 程度の比較的長いオーディオ信号を入力していると考えられる。
元論文ではおそらく USM の conformer block の最大入力長である 30 [sec] が入力長として用いられていることが示唆されているが、これはかなり大きなメモリを消費し多くの人にとって学習が困難なため、本レポジトリのサンプルコードでは 10 [sec] の入力をデフォルトとしている。
まとめると、各学習パートの入力信号は以下の通りとなる。

| Stage | Module                | Input length | Input audio | Purpose                                      |
|-------|-----------------------|--------------|------|----------------------------------------------|
| 1     | Feature Cleaner       | 10 sec  | Noisy speech | Clean noisy SSL features (main restoration) |
| 2     | WaveFit Pre-training  | 0.6 sec | Clean speech | Pre-train vocoder with clean features |
| 3     | WaveFit Fine-tuning   | 10 sec  | Noisy speech | Fine-tune vocoder with restored features |

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
1. "Miipher: A Robust Speech Restoration Model Integrating Self-Supervised Speech and Text Representations", Y. Koizumi, WASPAA 2021
1. "WaveFit: An Iterative and Non-autoregressive Neural Vocoder based on Fixed-Point Iteration", Y. Koizumi et al., IEEE SLT, 2022
1. "Google USM: Scaling Automatic Speech Recognition Beyond 100 Languages", Y. Zhang et al., Arxiv, 2023
1. "LibriTTS-R: A Restored Multi-Speaker Text-to-Speech Corpus", Yuma Koizumi et al., INTERSPEECH 2023.
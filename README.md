# ScrabbleGAN - Handwritten Text Generation

A PyTorch implementation of the [ScrabbleGAN: Semi-Supervised Varying Length Handwritten Text Generation](https://arxiv.org/abs/2003.10557) paper.

The main code was taken from [amzn/convolutional-handwriting-gan](https://github.com/amzn/convolutional-handwriting-gan) and [arshjot/ScrabbleGAN](https://github.com/arshjot/ScrabbleGAN).

## Quick setup and start

- Nvidia drivers >= 470, CUDA >= 11.4
- [Docker](https://docs.docker.com/engine/install/ubuntu/), [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

The provided [Dockerfile](Dockerfile) is supplied to build an image with CUDA support and cuDNN.

### Preparations

- Clone the repo.
- Download and extract dataset to the `data/` folder.
- `sudo make all` to build a docker image and create a container.
  Or `sudo make all GPUS=device=0 CPUS=10` if you want to specify gpu devices and limit CPU-resources.

If you don't want to use Docker, you can install dependencies via requirements.txt

### Preprocess data

The dataset should contain a folder with text images and csv file with annotations. The csv file should consist of two columns: "filename" with the relative path to the images (folder-name/image-name.png), and "text"-column with the image transcription.

To preprocess data and to create pickle:

```bash
python prepare_data.py \
	--data_csv_path data/hkr/folder-name.csv \
	--output_pkl_name data/folder-name_data.pkl
```

You can pass data_csv_path argument multuple times if you have a lot of csv files.

### Train

```bash
python train.py \
	--data_pkl_path data/folder-name_data.pkl \
	--unlabeled_pkl_path data/folder-name_unlabeled_data.pkl \
	--lexicon_path path/to/lexicon.txt
```

- unlabeled_pkl_path - path to unlabeled data, if you need to train model with images without transcriptions (these images will be used to train discriminator only).
- lexicon_path - the path to the txt-file with text in each line.

### Generate images

```bash
python generate_images.py \
	--checkpoint_path weights/model_checkpoint.pth.tar \
	--char_map_path data/folder-name_data.pkl \
	--num_imgs 10000 \
	--lexicon_path path/to/lexicon.txt \
	--output_path generated_synth/test
```

- char_map_path - path to labeled pickle data, that contains the alphabet of the data (char_map).
- lexicon_path - the path to the txt-file with text in each line. It may differ from the lexicon used for training the model.

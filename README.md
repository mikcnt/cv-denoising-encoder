# Denoising of path traced images using Deep Learning
Project for the [Computer Vision](http://visionlab.di.uniroma1.it/) course, Sapienza University.

The aim of this project is to build a Pytorch model able to denoise images. We're particularly interested in the noise produced by path-tracing ([this](https://youtu.be/frLwRLS_ZR0) is a cool video from Disney explaining this process, if you've never come across those words!).

One important part of this project was the investigation of the rendering noise. Is there a way to algorithmically recrate this noise? Great question. We've made use of simple Gaussian noise along with a revisited version of salt and pepper noise to solve this task.
## Usage
### Dependencies
In the repository, it is included `requirements.txt`, which consists in a file containing the list of items to be installed using conda, like so:

`conda install --file requirements.txt`

Once the requirements are installed, you shouldn't have any problem when executing the scripts. Consider also creating a new environment, so that you don't have to worry about what is really needed and what not after you're done with this project. With conda, that's easily done with the following command:

`conda create --name <env> --file requirements.txt`

where you have to replace `<env>` with the name you want to give to the new environment.

### Data structure 🗄️
To train the model from scratch, it is mandatory to have a data directory in which the files are organized as follows:
```
├── train
│   ├── 1.jpg
│   ├── ...
│   ├── 2.jpg
│   └── 3.jpg
└── test
    ├── 7.jpg
    ├── ...
    ├── 8.jpg
    └── 9.jpg
```
### Training 🏋️
Once you have the files well organized, you can start the training directly from command line. For example, to select batch size of 8, training and testing set in the directory data, run:

```shell
$ python main.py --batch_size 8 --data_path data
```

Apart from this super simple example, there are quite a few parameters that can be set. It is possible to resume last checkpoint, use drive as storage (e.g., if training on Colab), etc.
```shell
$ python main.py --h
usage: main.py [-h] [--model_checkpoint MODEL_CHECKPOINT] [--resume_last]
               [--batch_size BATCH_SIZE] [--epochs EPOCHS]
               [--learning_rate LEARNING_RATE] [--data_path DATA_PATH]
               [--use_drive]

Arguments parser

optional arguments:
  -h, --help            show this help message and exit
  --model_checkpoint MODEL_CHECKPOINT
                        path to .pth file checkpoint of the model (default:
                        none)
  --resume_last         use this flag to resume the last checkpoint of the
                        model
  --batch_size BATCH_SIZE
                        batch size (default: 8)
  --epochs EPOCHS       number of epochs (default: 500)
  --learning_rate LEARNING_RATE
                        learning rate (default 0.1)
  --data_path DATA_PATH
                        dataset path
  --use_drive           use this flag to save checkpoint on drive
```

### What about GANs? 🔫
This repository is strictly related to its [GAN twin](https://github.com/mikcnt/cv-denoising-gan), where there is some (very similar) code to train the model using a Generative Adversarial Network. Go check it out, if you're interested!
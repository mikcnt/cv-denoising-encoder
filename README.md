# Denoising of path traced images using Deep Learning
Project for the Computer Vision course, Sapienza University.

The aim of this project is to build a Pytorch model able to denoise images. We're particularly interested in the noise produced by path-tracing ([this](https://youtu.be/frLwRLS_ZR0) is a cool video from Disney explaining this process, if you've never come across those words!).

One great part of this process was the investigation of the rendering noise. Is there a way to algorithmically recrate this noise? Great question. We've made use of simple Gaussian noise, and some revisited version of salt and pepper noise to solve this task.

## Usage
### Dependencies
In the repository, it is included `requirements.txt`, which consists in a file containing the list of items to be installed using conda, like so:

`conda install --file requirements.txt`

Once the requirements are installed, you shouldn't have any problem when executing the scripts. Consider also creating a new environment, so that you don't have to worry about what is really needed and what not after you're done with this project. With conda, that's easily done with the following command:

`conda create --name <env> --file requirements.txt`

where you have to replace `<env>` with the name you want to give to the new environment.

### Data structure
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
### Training
Once you have the files well organized, you can start the training directly from command line, as shown in the following example:

```shell
$ python3 main.py --batch_size 8 --data_path data --
```
# Multi-retinal Disease Detection

## Deep Learning 2025 Final Project

Transfer Learning for Multi-label Medical Image Classification

*By Matt Stirling*

## Running the train script

To run the train script `matti.py`, please 

### Preparing dataset

Download resources from the [kaggle competition](https://www.kaggle.com/competitions/final-project-deep-learning-fall-2025/data).
Then, ensure the images and labels (csv files) are in the following format:

```
|--- ODIR_dataset/  # in same dir as matti.py
|    |--- images/       # images dir
|    |--- labels/       # all relevant csv files (train.csv, etc)
```

### Python environment

Ensure you are running a Python 3 environment with the following packages:

```
torch
torchvision
sklearn
pandas
pillow
```

### Options

Use `-h` to see list of all options (`python matti.py -h`). 


## Use of AI

- Help with freezing only backbone of model. 

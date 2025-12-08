# Deep Learning (2025) Final Project: *Multi-retinal Disease Detection*

*Transfer Learning for Multi-label Medical Image Classification.*

This project seeks to use transfer learning to fine-tune [ResNet18](https://arxiv.org/abs/1512.03385) and [EffecientNet_b0](https://arxiv.org/abs/1905.11946) models on the relatively small [ODIR](https://arxiv.org/abs/2102.07978) dataset. The idea is that models trained on large datasets can be leveraged for fine-tuning on smaller datasets. 

*By Matt Stirling*


## Running

To run the train script `matti.py`, please do the following:


### 1. Preparing dataset

Download resources from the [kaggle competition](https://www.kaggle.com/competitions/final-project-deep-learning-fall-2025/data).
Then, ensure the images and labels (csv files) are in the following format (relative to project root):

```
|--- ODIR_dataset/
|    |--- images/           # images dir
|    |--- labels/           # all relevant csv files (train.csv, etc)
|--- pretrained_backbone/   # the two pretrained backbone .pt files
```

### 2. Python environment

Ensure you are running a Python 3 environment with the following packages:

```
torch
torchvision
sklearn
pandas
pillow
```

### 3. Running

Select the mode (first unnamed argument) as either `train|test|predict|none`

**eg: Select backbone and train (simple)**

```
python matti.py train --backbone effnet --epochs 5 --save_name "example/effnet_5ep"
```

**eg: Select checkpoint and predict**

```
python matti.py predict --load_checkpoint "example/effnet_5ep"
```

**Note:** You can use mode `none` to just load model without further tasks. 

**Important options:**

| OPTION | DESCRIPTION |
|--|--|
| `--backbone` | select backbone |
| `--load_checkpoint` | load fine-tuning checkpoint (for testing, prediction, or further training) |
| `--epochs` | number of epochs to train for |
| `--ft_mode` | fine-tuning mode (classifier or all) |
| `--loss_fn` | loss function to use |
| `--lr` | learning rate |
| `--optimizer` | optimizer (eg. `adam`) |
| `--attention` | attention mechanism to use |

Use `-h` to see list of all options (eg. `python matti.py -h`).


## Use of AI

During the writing of this training script, I used ChatGPT (gpt-5 or lower) to help with the following tasks:

- Freezing only backbone of model. 
- Loading optimizers generically (from argparse)
- 

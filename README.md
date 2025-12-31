# Deep Learning (2025) Final Project: *Multi-label Retinal Disease Detection*

*Transfer Learning for Multi-label Medical Image Classification.*

This project seeks to use transfer learning to fine-tune [ResNet18](https://arxiv.org/abs/1512.03385) and [EffecientNet_b0](https://arxiv.org/abs/1905.11946) models on the relatively small [ODIR](https://arxiv.org/abs/2102.07978) dataset. The idea is that models trained on large datasets can be leveraged for fine-tuning on smaller datasets. 

*By Matt Stirling*


## Running

To run the train script `matti.py`, please do the following:


### 1. Preparing dataset

Download resources from the [kaggle competition](https://www.kaggle.com/competitions/final-project-deep-learning-fall-2025/data).
Then, ensure the images and labels (csv files) are in the following format (relative to project root):

```
|-- ODIR_dataset/
|   |-- images/                     # images dir
|   |-- labels/                     # all relevant csv files (train.csv, etc)
|-- pretrained_backbone/            # the two pretrained backbone .pt files
|   |-- ckpt_efficientnet_ep50.pt
|   |-- ckpt_resnet18_ep50.pt
|--- matti.py                       # my script
```

If you wish to run backbones other than EffNet or ResNet (such as Swin Transformer Tiny), please consult `notebooks/pretrained_backbone_exported.ipynb` in the [GitHub repo](https://github.com/MattThePerson/DeepLearning2025_FinalProject). 


### 2. Python environment

Ensure you are running a Python 3 environment with the following packages:

```
torch
torchvision
tensorboard
torchsummary
scikit
pillow
pandas
matplotlib
ipykernel
jupyter
tqdm
```

### 3. Running

Select the mode (first unnamed argument) as either `train|test|predict|none`

**eg: Load backbone and do nothing**

```
python matti.py none --backbone effnet --ft_mode all
```

**eg: Load backbone and train**

```
python matti.py train --backbone effnet --epochs 5 --save_name "example/effnet_5ep" -htp
```

**eg: Select checkpoint and predict**

```
python matti.py predict --load_checkpoint "example/effnet_5ep"
```


#### Important options:

| OPTION | DESCRIPTION |
|--|--|
| `--backbone` | select backbone to load |
| `--save_name` | What to name best checkpoint (can include folders) |
| `--load_checkpoint` | load fine-tuned checkpoint (for testing, prediction, or further training) |
| `--epochs` | number of epochs to train for |
| `--ft_mode` | fine-tuning mode (`classifier` or `all`) |
| `--loss_fn` | loss function to use |
| `--lr` | learning rate |
| `--lr_final` | final learning rate multiplier |
| `--optimizer` | optimizer (eg. `adam`) |
| `--attention_mechanism` | attention mechanism to use (`SE` or `MHA`) |

Use `-h` to see list of all options (eg. `python matti.py -h`).


## Use of AI

During the writing of this training script, I used ChatGPT (gpt-5 or lower) to help with various tasks, including:

- Freezing only backbone of model
- Loading optimizers generically (from argparse)
- Fixing focal loss class
- Help with contructing attention augmented models

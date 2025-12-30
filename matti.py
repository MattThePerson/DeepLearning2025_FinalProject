"""
Code for training and fine-tuning. By Matt Stirling. 
"""
from typing import Any
import argparse

import os
import sys
import csv
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.ops as ops
from torchvision import transforms, models

from torch.utils.tensorboard.writer import SummaryWriter


DEVICE: torch.device


# ======================================================================================================================
# region Data Sets/Loaders
# ======================================================================================================================
class RetinaMultiLabelDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row.iloc[0])
        img = Image.open(img_path).convert("RGB")
        labels = torch.tensor(row[1:].values.astype("float32"))
        if self.transform:
            img = self.transform(img)
        return img, labels

class RetinaMultiLabelDataset_WithoutLabels(Dataset):
    def __init__(self, image_dir, transform=None):
        self.images = os.listdir(image_dir)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img_name


def get_dataloaders(dataset_path: str, img_size=256, batch_size=32):
    
    # transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # paths
    train_csv = os.path.join( dataset_path, "labels/train.csv" )
    val_csv   = os.path.join( dataset_path, "labels/val.csv" )
    test_csv  = os.path.join( dataset_path, "labels/offsite_test.csv" )
    
    train_image_dir = os.path.join( dataset_path, "images/train" )
    val_image_dir =   os.path.join( dataset_path, "images/val" )
    test_image_dir =  os.path.join( dataset_path, "images/offsite_test" )
    onsite_test_image_dir =  os.path.join( dataset_path, "images/onsite_test" )

    # dataset & dataloader
    train_ds =       RetinaMultiLabelDataset(train_csv, train_image_dir, transform)
    val_ds   =       RetinaMultiLabelDataset(val_csv, val_image_dir, transform)
    test_ds  =       RetinaMultiLabelDataset(test_csv, test_image_dir, transform)
    onsite_test_ds = RetinaMultiLabelDataset_WithoutLabels(onsite_test_image_dir, transform)

    num_workers = 4
    import platform
    if platform.system() == "Windows": num_workers = 0 # Stupid shit fucking windows

    train_loader =        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   =        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  =        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    onsite_test_loader  = DataLoader(onsite_test_ds, batch_size=1, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, onsite_test_loader


# ======================================================================================================================
# region ATTN MODELS
# ======================================================================================================================

from torchvision.models.efficientnet import EfficientNet, _efficientnet_conf


# Multihead Attention augmented EfficientNet

class EfficientNet_MHA(EfficientNet):
    """ 
    EfficientNet model augmented with a Multihead Attention module from `this paper
    <https://arxiv.org/abs/1706.03762>`_.  
    
    Args:
        num_heads (int): Number of heads to use in Multihead Attention layer. 
    """
    def __init__(
            self,
            num_heads: int=8,
            **kwargs: Any,
        ):
            super().__init__(**kwargs)
            fc: nn.Linear = self.classifier[1] # type: ignore[assignment]
            embed_dim = fc.in_features # same as output channels of self.features
            self.MHA = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
    
    def attention(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2) # (N, W*H, C) get correct shape for MHA
        x, _ = self.MHA(x, x, x)         # (N, W*H, C)
        x = torch.mean(x, 1)             # (N, C)
        return x
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """ Modified forward. Replaced avgpool and flatten layer with attention block."""
        x = self.features(x)    # (N, C, W, H)
        x = self.attention(x)   # (N, C)
        x = self.classifier(x)
        return x


def efficientnet_b0_MHA(
    num_heads: int=8,
    **kwargs: Any
) -> EfficientNet_MHA:

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)

    return EfficientNet_MHA(
        inverted_residual_setting=inverted_residual_setting,
        last_channel=last_channel,
        dropout=0.2,
        num_heads=num_heads,
        **kwargs,
    )


# Squeeze-and-Excitation augmented EfficientNet

class EfficientNet_SE(EfficientNet):
    """ 
    EfficientNet model augmented with a Squeeze-and-Excitation attention block
    (from https://arxiv.org/abs/1709.01507). 
    
    Args:
        reduction ratio (int): Determines number of squeeze channels by dividing number of input channels
    """
    def __init__(
            self,
            reduction_ratio: int = 16,
            **kwargs: Any,
        ):
            super().__init__(**kwargs)
            fc: nn.Linear = self.classifier[1] # type: ignore[assignment]
            input_channels = fc.in_features # same as output channels of self.features
            squeeze_channels = input_channels // reduction_ratio
            self.SE = ops.SqueezeExcitation(input_channels, squeeze_channels)
    
    def attention(self, x: torch.Tensor) -> torch.Tensor:
        x = self.SE(x)           # (N, C, W, H)
        return x
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """ Modified forward by adding attention module before avgpool """
        x = self.features(x)    # (N, C, W, H)
        x = self.attention(x)   # (N, C, W, H)
        x = self.avgpool(x)     # (N, C, 1, 1)
        x = torch.flatten(x, 1) # (N, C)
        x = self.classifier(x)
        return x


def efficientnet_b0_SE(
    reduction_ratio: int=16,
    **kwargs: Any,
) -> EfficientNet_SE:

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)

    return EfficientNet_SE(
        inverted_residual_setting=inverted_residual_setting,
        last_channel=last_channel,
        dropout=0.2,
        reduction_ratio=reduction_ratio,
        **kwargs,
    )



# ======================================================================================================================
# region BUILD MODEL
# ======================================================================================================================


def build_resnet(attn=None, num_classes=3):
    # models.resnet.ResNet
    if attn is None:
        model = models.resnet18(weights=None)
    else:
        raise Exception("No attention mechanisms implemented for ResNet")
        
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_efficientnet(attn=None, num_classes=3):
    # models.efficientnet.EfficientNet
    if attn is None:
        model = models.efficientnet_b0(weights=None)
    elif attn == "MHA":
        model = efficientnet_b0_MHA()
    elif attn == "SE":
        model = efficientnet_b0_SE()
    else: 
        raise Exception(f"No such attention mechanism for EfficientNet: {attn}")

    layer_fc: nn.Linear = model.classifier[1] # type: ignore[assignment]
    model.classifier[1] = nn.Linear(layer_fc.in_features, num_classes)
    return model


def build_swin_b(num_classes=3):
    # models.swin_transformer.SwinTransformer
    model = models.swin_b(
        weights = models.Swin_B_Weights.IMAGENET1K_V1,
    )
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model


def build_swin_t(num_classes=3):
    # models.swin_transformer.SwinTransformer
    model = models.swin_t(
        weights = models.Swin_T_Weights.IMAGENET1K_V1,
    )
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model


def build_vision_b(num_classes=3):
    # models.vision_transformer.VisionTransformer
    model = models.vit_b_16(
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1,
    )
    fc: nn.Linear = model.heads.head # type: ignore[assignment]
    model.heads.head = nn.Linear(fc.in_features, num_classes)
    return model



def build_model(backbone="resnet18", attn=None, num_classes=3):

    if backbone == "resnet18":
        model = build_resnet(attn, num_classes)
    elif backbone == "effnet":
        model = build_efficientnet(attn, num_classes)
    elif backbone == "swin_t":
        model = build_swin_t(num_classes)
    elif backbone == "swin_b":
        model = build_swin_b(num_classes)
    elif backbone == "vision":
        model = build_vision_b(num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    return model


def get_model(backbone="resnet18", attn=None, pretrained_params=None, freeze_backbone=False, num_classes=3):
    
    model = build_model(backbone, attn, num_classes)
    
    # pretrained params
    if pretrained_params is not None:
        print('loading params:', pretrained_params)
        state_dict = torch.load(pretrained_params, map_location="cpu")
        try:
            # TODO: load state with minimum number of incompatible layers
            #       (OR) add exception to attention mechanism models
            model.load_state_dict(state_dict)
        except:
            print(f"ERROR: Incompatible backbone ({backbone}) and params file ({pretrained_params})\n ...exiting")
            sys.exit(2)
        
    else:
        print('\033[33mImportant:\033[0m Not loading any params, training model from SCRATCH')
    
    # parameters freezing
    if freeze_backbone:
        print('FREEZING: freezing model backbone (non-Linear layers)')
        model = freeze_non_final_linear_layer(model)
    
    else:
        print('FREEZING: Unfreezing all layers')
        for p in model.parameters():
            p.requires_grad = True
    
    # print param amounts
    all_params, trainable_params = get_parameter_count(model)
    print('=====================')
    print('    LOADED MODEL')
    print('---------------------')
    print('backbone:', backbone)
    print('attn:', attn)
    print('pretrained params:', pretrained_params)
    print('parameter count:  {:_d}'.format(all_params))
    print('trainable params: {:_d}'.format(trainable_params))
    print('frozen params:    {:_d}'.format(all_params-trainable_params))
    print('=====================')
    
    return model


# ======================================================================================================================
# region HELPERS
# ======================================================================================================================

# TODO: remove, deprecated!
def freeze_non_linear_layers(model):
    """
    Freezes all non linear layers
    """
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze only Linear layers
    for m in model.modules():
        if isinstance(m, nn.Linear):
            for p in m.parameters():
                p.requires_grad = True
    return model


def freeze_non_final_linear_layer(model):
    """
    Freeze backbone and leave classifier (final linear layer) unfrozen. 
    """
    for p in model.parameters():
        p.requires_grad = False

    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m

    if last_linear is not None:
        for p in last_linear.parameters():
            p.requires_grad = True

    return model

def get_parameter_count(model): 
    all_params =       sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return all_params, trainable_params

def ensure_parent_exists(file: str):
    parent = os.path.dirname(file)
    if parent != '':
        os.makedirs(parent, exist_ok=True)

def display_loss(train_loss, prev_train_loss, val_loss, prev_val_loss):
    ANSI_reset = "\033[0m"
    ANSI_red = "\033[31m"
    ANSI_green = "\033[32m"
    train_diff = train_loss - prev_train_loss
    val_diff = val_loss - prev_val_loss
    train_diff_col = ANSI_green if (train_diff <= 0) else ANSI_red
    val_diff_col =   ANSI_green if (val_diff <= 0) else ANSI_red
    train_msg = f"train loss: {train_loss:.4f} ({train_diff_col}{train_diff:+.4f}{ANSI_reset})"
    val_msg =     f"val loss: {val_loss:.4f} ({val_diff_col}{val_diff:+.4f}{ANSI_reset})"
    print(f"  {train_msg:<35}   {val_msg}")

def save_test_results(df: pd.DataFrame, params: str|None, save_dir="test_results"):
    save_name = f"{save_dir}/test_results.csv"
    if params is not None:
        params = params.replace("\\", "/")
        substr = "checkpoints/"
        if substr in params:
            params = params.replace(substr, f"{save_dir}/")
        else:
            params = f"{save_dir}/{params}"
        save_name = params.replace('.pt', '.csv')
    ensure_parent_exists(save_name)
    print('saving test results to:', save_name)
    df.to_csv(save_name)

def hyperparams_to_string(a) -> str:
    s = ""
    if a.loss_fn != "bce":
        s += f"loss={a.loss_fn}"
    if a.attention_mechanism:
        s += f"_attn={a.attention_mechanism}"
    optim = f"optim={a.optimizer}_lr={a.lr}"
    if a.momentum != 0.0:
        optim += f"_mom={a.momentum}"
    if a.weight_decay != 0.0:
        optim += f"_dec={a.weight_decay}"
    if a.lr_final != 1.0:
        optim += f"_lrf={a.lr_final}"
    return f"{s}_{optim}_{a.epochs}ep"


# ======================================================================================================================
# region CUSTOM LOSS
# ======================================================================================================================

class MyFocalLossWithLogits(nn.Module):
    """
    *(Description from assignment) Focal Loss: A loss function designed to address class imbalance by downweighting
    easy examples and focusing training on hard, misclassified ones.*
    
    My custom implementation of Focal Loss (with logits). Had some help from gpt-5. 
    
    Implements the equation:

        FL(p_t) = - alpha_t * (1-p_t)^gamma * log(p_t)
    
    Forward: shape of logits and targets is identical. 
    """
    def __init__(self, gamma: float=2.0):
        super().__init__()
        self.gamma = gamma
    
    def get_pt_softmax(self, logits, targets):
        """ Returns the models estimated probability for the correct class """
        probs = F.softmax(logits, dim=1)
        p_t = (probs * targets).mean(dim=1)
        return p_t
    
    def get_pt_bce(self, logits, targets):
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce_loss)
        F.binary_cross_entropy_with_logits
        return p_t
    
    def forward(self, logits, targets):
        p_t = self.get_pt_bce(logits, targets)
        p_t = torch.clamp(p_t, min=1e-7, max=1.0)  
        # focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        focal_loss = - (1-p_t)**self.gamma * torch.log(p_t)
        F.nll_loss
        return focal_loss.mean()



# Modified version of Class Balanced Loss function found here:
#   https://github.com/vandit15/Class-balanced-loss-pytorch/
class ClassBalancedBCEWithLogitsLoss(nn.Module):
    """
    Class-Balanced BCE loss using the formula:
    
    ((1 - beta) / (1 - beta^n_y)) * BCEWithLogitsLoss(logits, targets)
    """
    def __init__(self, beta: float=0.999):
        super().__init__()
        self.beta = beta

    def get_weights(self, targets):
        C = targets.shape[1]
        samples_per_cls = targets.sum(dim=0)

        beta_per_cls = torch.pow(self.beta, samples_per_cls)
        denom = 1.0 - beta_per_cls
        weights = (1.0 - self.beta) / torch.clamp(denom, min=1e-7)
        weights = weights / torch.sum(weights) * C # normalize mean to 1

        # # class weights to sample weights
        weights = (weights.unsqueeze(0) * targets).sum(dim=1)
        weights = weights.unsqueeze(1)

        return weights

    def forward(self, logits, targets):
        """ 
        logits:  (N, C)
        targets: (N, C)
        """
        weights = self.get_weights(targets)
        cb_loss = F.binary_cross_entropy_with_logits(
            input=logits,
            target=targets.float(),
            weight=weights,
        )
        return cb_loss



# ======================================================================================================================
# region predict
# ======================================================================================================================
def predict(
        model: nn.Module,
        loader: DataLoader,
        csv_path="onsite_test_submission.csv",
    ):
    
    model.eval()
    data = []
    print(f'generating predictions for {len(loader.dataset)} images') # type: ignore
    with torch.no_grad():
        for img, img_name in tqdm(loader):
            img_name = img_name[0]
            img = img.to(DEVICE)
            output = model(img)[0]
            probs = torch.sigmoid(output).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            data_line = [img_name]
            data_line.extend(preds)
            data.append(data_line)
    
    # write to csv
    if not csv_path.endswith(".csv"): csv_path += ".csv"
    ensure_parent_exists(csv_path)
    print(f'writing predictions to {csv_path}')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id","D","G","A"])
        writer.writerows(data)

# ======================================================================================================================
# region test
# ======================================================================================================================
def test(
        model: nn.Module,
        loader: DataLoader,
    ):

    print(f'Testing model on {len(loader.dataset)} images') # type: ignore

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="TESTING", colour="magenta"):
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            
    y_true = np.array(y_true) #torch.tensor(y_true).numpy()
    y_pred = np.array(y_pred) #torch.tensor(y_pred).numpy()

    # compute metrics
    disease_names = ["DR", "Glaucoma", "AMD"]
    results_data = []
    
    for i, disease in enumerate(disease_names):  # compute metrics for every disease
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        acc =       accuracy_score(y_t, y_p)
        precision = precision_score(y_t, y_p, average="binary", zero_division=0)
        recall =    recall_score(y_t, y_p, average="binary", zero_division=0)
        f1 =        f1_score(y_t, y_p, average="binary", zero_division=0)
        kappa =     cohen_kappa_score(y_t, y_p)

        results_data.append([disease, acc, precision, recall, f1, kappa])

    results = pd.DataFrame(
        data=results_data, 
        columns=["Category", "Accuracy", "Precision", "Recall", "F1-score", "Kappa"],
    ).set_index("Category")

    results = results.T
    disease_occurances = y_true.sum(axis=0)
    results["Average"] = np.average(results.values, axis=1, weights=disease_occurances)
    print("========================")
    print("DISEASE SPECIFIC METRICS:\n")
    print(results.T)
    print()
    
    return results.T


# ======================================================================================================================
# region train
# ======================================================================================================================
def train(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        epochs=10,
        save_name="best.pt",
        checkpoints_dir="checkpoints",
        runs_dir="runs",
        save_csv=True,
        lr0=0.001,
        lr_final=1.0,
    ) -> str:

    writer = SummaryWriter(f".tensorboard/{save_name.replace('.pt', '')}")

    # csv writing
    run_csv_path = os.path.join( runs_dir, save_name.replace(".pt", ".csv") )
    ensure_parent_exists(run_csv_path)
    def write_to_csv(msg, mode="w"):
        if save_csv:
            with open(run_csv_path, mode) as f:
                f.write(msg)
    write_to_csv("epoch,train_loss,val_loss,acc,precision,recall,f1\n")
    
    save_name = os.path.join( checkpoints_dir, save_name )
    ensure_parent_exists(save_name)
    
    # ITERATE
    lr_delta = (lr0 - lr0 * lr_final) / (epochs-1)
    best_val_loss = float("inf")
    prev_train_loss, prev_val_loss = float("inf"), float("inf")
    for epoch in range(epochs):
        
        # determine learning rate
        lr = lr0 - lr_delta * epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'using lr={lr:.7f}')
        
        # train
        model.train()
        train_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"EPOCH {epoch+1}/{epochs}", colour="cyan"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset) # type: ignore
        
        # validation
        model.eval()
        val_loss = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds)
        val_loss /= len(val_loader.dataset) # type: ignore
        
        # metrics
        acc =       accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall =    recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 =        f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"    ..saving best checkpoint to: {save_name}")
            torch.save(model.state_dict(), save_name)
        
        # display & write
        display_loss(train_loss, prev_train_loss, val_loss, prev_val_loss)
        prev_train_loss, prev_val_loss = train_loss, val_loss
        write_to_csv(f"{epoch+1},{train_loss:.5f},{val_loss:.5f},{acc:.5f},{precision:.5f},{recall:.5f},{f1:.5f}\n","a")
        writer.add_scalar("loss/train", train_loss, epoch+1)
        writer.add_scalar("loss/val", val_loss, epoch+1)
        writer.add_scalar("metrics/acc", acc, epoch+1)
        writer.add_scalar("metrics/precision", precision, epoch+1)
        writer.add_scalar("metrics/recall", recall, epoch+1)
        writer.add_scalar("metrics/f1", f1, epoch+1)

    writer.close()
    return save_name


# ======================================================================================================================
# region MAIN
# ======================================================================================================================
def main(
        mode = "train",
        backbone = "resnet18",
        dataset_path = "dataset",
        pretrained_params: str|None = None,
        save_name = "best.pt",
        checkpoints_dir = "checkpoints",
        runs_dir = "runs",
        freeze_backbone = True, # freeze non-linear layers
        loss_fn = nn.BCEWithLogitsLoss,
        attention_mechanism = None,
        predict_csv = "onsite_test_submission.csv",
        epochs = 10,
        opt_class = optim.Adam,
        opt_kwargs = {},
        batch_size=32, img_size=256, num_classes=3,
        lr_final = 1.0, # multiplier for final learning rate target
    ):
    
    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using device:', DEVICE)
    
    train_loader, val_loader, test_loader, onsite_test_loader = get_dataloaders(dataset_path, img_size, batch_size)

    print('Building model')
    model = get_model(
        backbone = backbone,
        attn = attention_mechanism,
        pretrained_params = pretrained_params,
        freeze_backbone = freeze_backbone,
        num_classes = num_classes,
    ).to(DEVICE)
    
    # MODE
    match mode:
        case "train": # - train -------
            lr0 = opt_kwargs.pop('lr', 0.001)
            
            optimizer = opt_class(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr0,
                **opt_kwargs,
            )

            print(f'Training {backbone} for {epochs} epochs')
            print('loss function:', loss_fn)
            best_ckpt = train(
                model,
                train_loader,
                val_loader,
                optimizer,
                loss_fn,
                epochs=epochs,
                save_name=save_name,
                checkpoints_dir=checkpoints_dir,
                runs_dir=runs_dir,
                lr0=lr0,
                lr_final=lr_final,
            )
            print(f"Loading best checkpoint '{best_ckpt}' and testing model")
            model.load_state_dict(torch.load(best_ckpt, map_location="cpu"))
            results_df = test(
                model,
                test_loader,
            )
            save_test_results(results_df, save_name)
        
        case "test":  # - test ---------
            results_df = test(
                model,
                test_loader,
            )
            save_test_results(results_df, pretrained_params)
        
        case "predict": # - predict ---
            print("Predicting ...")
            predict(
                model,
                onsite_test_loader,
                csv_path=predict_csv,
            )
        
        case "none": # - none ---------
            print("Just a test, nothing to see here")
        
        case _:
            print("oh no, no mode is matched??")



# ======================================================================================================================
# region CLI HELPERS
# ======================================================================================================================

LOSS_FUNCS = {
    "bce":              nn.BCEWithLogitsLoss(),
    "focal":            MyFocalLossWithLogits(),
    "class_balanced":   ClassBalancedBCEWithLogitsLoss(),
}

# TODO: add descriptions
ATTENTION_MECHANISMS = {
    "SE": "Squeeze-and-Excitation: Attention block which adaptively recalibrates channel-wise feature responses by \
                                   explicitly modelling interdependencies between channels",
    "MHA": "Multi-head Attention: Attention mechanism described in `Attention is all you need` paper.",
}

PRETRAINED_BACKBONES = {
    'resnet18':         './pretrained_backbone/ckpt_resnet18_ep50.pt',
    'effnet':           './pretrained_backbone/ckpt_efficientnet_ep50.pt',
    'swin_t':           './pretrained_backbone/ckpt_swin_t.pt',
    'vision':           './pretrained_backbone/ckpt_visionTransformer.pt',
}

OPTIMIZERS = {
    "sgd":  (optim.SGD,  {"lr", "momentum", "weight_decay"}),
    "adam": (optim.Adam, {"lr", "weight_decay"}),
    "adamw": (optim.AdamW, {"lr", "weight_decay"}),
}


def get_checkpoints(root: Path|str) -> list[str]:
    root = Path(root)
    paths = [ str(root / p.relative_to(root)) for p in root.rglob("*.pt") ]
    paths = [ pth.replace("\\", "/") for pth in paths ]
    return paths


def handle_args(
    args: argparse.Namespace,
):
    # backbone & checkpoint
    checkpoints = get_checkpoints(args.checkpoints_dir)
    if args.list_checkpoints:
        print(f"Checkpoints detected in '{args.checkpoints_dir}':")
        for i, f in enumerate(checkpoints):
            print(f"  {i+1:>3}: {f}")
        sys.exit(0)
    
    params_file = None
    
    if args.load_checkpoint:
        filters = [ f.lower().strip() for f in args.load_checkpoint.split(",") ]
        filtered_ckpts = checkpoints
        for fil in filters:
            filtered_ckpts = [ f for f in filtered_ckpts if fil in str(f).lower() ]
        
        if len(filtered_ckpts) == 0:
            print(f"No checkpoint found for: '{args.load_checkpoint}'. Please select from the following:")
            for i, f in enumerate(checkpoints):
                print(f"  {i+1:>3}: {f}")
            sys.exit(2)
        
        elif len(filtered_ckpts) > 1:
            print(f"Got multiple checkpoints, please select from the following:")
            for i, f in enumerate(filtered_ckpts):
                print(f"  {i+1:>3}: {f}")
            sys.exit(2)
        
        params_file = str(filtered_ckpts[0])
        print('[PARAMS] using checkpoint:', params_file)
        
        # autodetect backbone
        for bb in PRETRAINED_BACKBONES.keys():
            if bb.lower() in params_file:
                print(f"[AUTODETECT] Detected backbone from checkpoint filename (overriding --backbone)")
                args.backbone = bb
                break

        # autodetect attention
        for attn in ATTENTION_MECHANISMS.keys():
            if f"attn={attn}" in params_file:
                print(f"[AUTODETECT] Detected attention mechanism from filename: {attn}")
                args.attention_mechanism = attn
                break
        
    elif args.no_pretrained_params:
        print("[PARAMS] not using any pretrained/fine-tuned parameters")
    
    else:
        params_file = PRETRAINED_BACKBONES.get(args.backbone.lower(), None)
        if params_file is None:
            print(f"No such backbone: '{args.backbone}': Please select from the following:")
            for i, (k, v) in enumerate(PRETRAINED_BACKBONES.items()):
                print(f"  {i+1:>3}: {k:<12} ({v})")
            sys.exit(2)
        print('[PARAMS] using pretrained backbone:', params_file)
    
    # optimizer
    opt_class, valid_keys = OPTIMIZERS.get(args.optimizer, (None, None))
    if opt_class is None:
        print(f"No such optimizer: {args.optim}.\nAvailable optimizers:")
        for i, (k, v) in enumerate(OPTIMIZERS.items()):
            print(f"  {i+1:>3}: {k:<12} hparams={{{v}}}")
        sys.exit(2)
    opt_kwargs = { k: getattr(args, k) for k in valid_keys }
    print(f"[OPTIMIZER] Using optimizer: {opt_class} with hyperparams:")
    for k, v in opt_kwargs.items():
        print(f"{k:>15}: {v}")
    print()
    
    # loss fn
    if args.loss_fn not in LOSS_FUNCS:
        if args.loss_fn != "help":
            print("[ERROR] No such loss function:", args.loss_fn)
        print("Available loss functions:")
        for i, (k, v) in enumerate(LOSS_FUNCS.items()):
            print(f"  {i+1:>3}: {k:<18} ({v})")
        sys.exit(2)
    
    # attention
    if args.attention_mechanism is not None and args.attention_mechanism not in ATTENTION_MECHANISMS:
        if args.attention_mechanism != "help":
            print("[ERROR] No such attention mechanism:", args.attention)
        print("Available attention mechanisms:")
        for i, (k, v) in enumerate(ATTENTION_MECHANISMS.items()):
            print(f"  {i+1:>3}: {k:<18} ({v})")
        sys.exit(2)
    
    # autoname predict_csv
    if args.predict_csv is None and params_file is not None:
        params_file_linux = params_file.replace("\\", "/")
        params_file_linux = params_file_linux.replace("checkpoints/", "predictions/")
        args.predict_csv = params_file_linux.replace(".pt", "") + ".csv"
    
    # save name
    savename = args.save_name
    hyper_str = hyperparams_to_string(args)
    
    if savename is None:
        savename = f"best"
    if args.hyperparams_to_name:
        savename += f"_{args.backbone}_{args.ft_mode}_({hyper_str})"
    if not savename.endswith(".pt"):
        savename += ".pt"
    
    if os.path.exists( os.path.join(args.checkpoints_dir, savename) ) and args.mode == "train":
        if input(f"\033[31mImportant:\033[0m Checkpoint file ('{savename}') already exists. \
                   Are you sure you want to replace it?\n ('y' or 'yes') > "
                ).lower() not in ["y", "yes", "yeahboii"]:
            print(" ..quitting\n")
            sys.exit(0)

    return args, params_file, savename, opt_class, opt_kwargs


# ======================================================================================================================
# region CLI
# ======================================================================================================================

if __name__ == "__main__":

    # ARGS
    # ------------------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument("mode", nargs="?", choices=["train", "test", "predict", "none"], default="none")
    parser.add_argument('--dataset_path', default="./ODIR_dataset", help='Path to dataset root')
    parser.add_argument('--backbone', '-b', choices=["resnet18", "effnet", "swin_t", "vision"], default="resnet18",
                            help='Which model to use as backbone (else: detects ackbone from --load_checkpoint)')
    parser.add_argument('--no_pretrained_params', '-npp', action='store_true',
                            help="Don't load any params (re-initialize weights, train from scratch)")
    parser.add_argument('--load_checkpoint', '-ckp',
                            help="Path to checkpoint to load (relative to `checkpoints/`) (else: load pretrained \
                                  backbone from `pretrained_backbone/`)")
    
    # train args
    parser.add_argument('--save_name', '-sn', help="Path to save best checkpoint (in checkpoints/)")
    parser.add_argument('--ft_mode', choices=["classifier", "all"], default="all",
                        help="Fine-tuning mode: which params to unfreeze")
    parser.add_argument('--loss_fn', default="bce", choices=["bce", "focal", "class_balanced"],
                        help="Loss function to use during training")
    parser.add_argument('--attention_mechanism', '-attn', choices=["SE", "MHA"], default=None,
                            help="Attention mechanism to use (use help to list options)")
    parser.add_argument('--batch_size', type=int, default=32)
    
    # hyperparams
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--optimizer', type=str, choices=["sgd", "adam", "adamw"], default="adam")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--momentum', type=float, default=0.0, help="Only for SGD optimizer")

    # advanced train args
    parser.add_argument('--lr_final', type=float, default=1.0,
                        help="lr at final epoch (eg. 1e-2)")
    parser.add_argument('--unfreeze_backbone_after', type=int, default=-1,
                        help="Epochs after which to unfreeze the backbone")
    parser.add_argument('--unfreeze_backbone_lr_mult', type=float, default=0.333,
                        help="Learning rate multiplier after unfreezing backbone")

    # misc
    parser.add_argument('--predict_csv', help="Path to csv to save output in predict mode")
    parser.add_argument('--checkpoints_dir', default="checkpoints", 
                            help="Folder to save and load checkpoints (default `checkpoints/`)")
    parser.add_argument('--list_checkpoints', '-ls', action="store_true", help="List all detected checkpoints")
    parser.add_argument('--hyperparams_to_name', '-htn', action='store_true', help="")

    parser.add_argument('--num_classes', type=int, default=3,
                            help="Number of classes we want to detect (changes shape of classifier). \
                                  Note: if not 3, pretrained backbone won't load.")
    
    args = parser.parse_args()
    
    # handle args
    args, params_file, savename, opt_class, opt_kwargs = handle_args(args)
    
    
    # MAIN
    # ------------------------------------------------------------------------------------------------------------------
    
    try:
        main(
            mode = args.mode,
            dataset_path = args.dataset_path,
            backbone = args.backbone,
            pretrained_params = params_file,
            save_name = savename,
            checkpoints_dir = args.checkpoints_dir,
            runs_dir = "runs",
            freeze_backbone = (args.ft_mode == "classifier"),
            loss_fn = LOSS_FUNCS[args.loss_fn],
            attention_mechanism = args.attention_mechanism,
            predict_csv = args.predict_csv,
            epochs = args.epochs,
            opt_class = opt_class,
            opt_kwargs = opt_kwargs,
            # lr = args.lr,                                       # default: 1e-5
            # batch_size = args.batch_size,                       # default: 32
            num_classes = args.num_classes,
            lr_final = args.lr_final,
            # unfreeze_backbone_after = args.unfreeze_backbone_after,
        )
    except KeyboardInterrupt:
        print("\n  ..caught KeyboardInterrupt, stopping\n")

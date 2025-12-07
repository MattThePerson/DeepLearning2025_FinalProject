"""
Code for training and fine-tuning. By Matt Stirling. 
"""
import os
import sys
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

from tqdm import tqdm
import csv

# ========================
# region Data Sets/Loaders
# ========================
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




# ========================
# region BUILD MODEL
# ========================

def build_model(backbone="resnet18", num_classes=3):

    if backbone == "resnet18":
        model = models.resnet18(weights=None) #"IMAGENET1K_V1") # TODO: should this be None??
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "effnet":
        model = models.efficientnet_b0(weights=None) #"IMAGENET1K_V1")
        layer_fc: nn.Linear = model.classifier[1] # type: ignore[assignment]
        model.classifier[1] = nn.Linear(layer_fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    
    return model


def get_model(backbone="resnet18", pretrained_params=None, freeze_backbone=False, num_classes=3):
    
    model = build_model(backbone, num_classes)
    
    # parameters freezing
    if freeze_backbone:
        print('FREEZING: freezing model backbone (non-Linear layers)')
        model = freeze_non_linear_layers(model)
    
    else:
        print('FREEZING: Unfreezing all layers')
        for p in model.parameters():
            p.requires_grad = True
    
    # pretrained params
    if pretrained_params is not None:
        print('loading params:', pretrained_params)
        state_dict = torch.load(pretrained_params, map_location="cpu")
        try:
            model.load_state_dict(state_dict)
        except:
            print(f"ERROR: Incompatible backbone ({backbone}) and params file ({pretrained_params})\n ...exiting")
            sys.exit(2)
        
    else:
        print('\033[33mImportant:\033[0m Not loading any params, training model from SCRATCH')
    
    # print param amounts
    all_params, trainable_params = get_parameter_count(model)
    print('=====================')
    print('    LOADED MODEL')
    print('---------------------')
    print('backbone:', backbone)
    print('pretrained params:', pretrained_params)
    print('parameter count:  {:_d}'.format(all_params))
    print('trainable params: {:_d}'.format(trainable_params))
    print('frozen params:    {:_d}'.format(all_params-trainable_params))
    print('=====================')
    
    return model


# ========================
# region HELPERS
# ========================

def freeze_non_linear_layers(model):
    """
    Freeze backbone and leave classifier (linear layers) unfrozen. 
    """
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze only Linear layers
    for m in model.modules():
        if isinstance(m, nn.Linear):
            for p in m.parameters():
                p.requires_grad = True
    return model

def get_parameter_count(model): 
    all_params =       sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return all_params, trainable_params

def ensure_parent_exists(file: str):
    os.makedirs(os.path.dirname(file), exist_ok=True)

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
    

# ========================
# region CUSTOM LOSS
# ========================

def FocalLoss(x):
    """
    Focal Loss: A loss function designed to address class imbalance by downweighting easy examples and focusing
    training on hard, misclassified ones.
    """
    raise NotImplementedError()

def ClassBalancedLoss(x):
    """
    Class-Balanced Loss: Re-weight the BCE loss according to class frequency. This is a common method for handling
    class imbalance.
    """
    raise NotImplementedError()


# ========================
# region predict
# ========================
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

# ========================
# region test
# ========================
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
    disease_counts = []
    
    for i, disease in enumerate(disease_names):  # compute metrics for every disease
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        disease_counts.append(y_t.sum())
        acc =       accuracy_score(y_t, y_p)
        precision = precision_score(y_t, y_p, average="macro", zero_division=0)
        recall =    recall_score(y_t, y_p, average="macro", zero_division=0)
        f1 =        f1_score(y_t, y_p, average="macro", zero_division=0)
        kappa =     cohen_kappa_score(y_t, y_p)

        results_data.append([disease, acc, precision, recall, f1, kappa])

    disease_weights = np.array(disease_counts) / np.sum(disease_counts)

    results = pd.DataFrame(
        data=results_data, 
        columns=["Disease", "Accuracy", "Precision", "Recall", "F1-score", "Kappa"],
    ).set_index("Disease")

    results = results.T
    results["Average"] = np.average(results.values, axis=1, weights=disease_weights)
    print("========================")
    print("DISEASE SPECIFIC METRICS:\n")
    print(results.T)
    print()
    
    return results


# ========================
# region train
# ========================
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
    ) -> str:

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
    best_val_loss = float("inf")
    prev_train_loss, prev_val_loss = float("inf"), float("inf")
    for epoch in range(epochs):
        
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
                y_true.extend(labels.numpy())
                y_pred.extend(preds)
        val_loss /= len(val_loader.dataset) # type: ignore
        
        # metrics
        acc =       accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall =    recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 =        f1_score(y_true, y_pred, average="macro", zero_division=0)

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"    ..saving best checkpoint to: {save_name}")
            torch.save(model.state_dict(), save_name)
        
        # display & csv
        display_loss(train_loss, prev_train_loss, val_loss, prev_val_loss)
        prev_train_loss, prev_val_loss = train_loss, val_loss
        write_to_csv(f"{epoch+1},{train_loss:.5f},{val_loss:.5f},{acc:.f},{precision:.5f},{recall:.5f},{f1:.5f}\n", "a")

    return save_name


# ========================
# region MAIN
# ========================
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
        attention = None,
        predict_csv = "onsite_test_submission.csv",
        epochs=10, batch_size=32, lr=1e-4, img_size=256, num_classes=3
    ):
    
    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using device:', DEVICE)

    train_loader, val_loader, test_loader, onsite_test_loader = get_dataloaders(dataset_path, img_size, batch_size)

    print('Building model')
    model = get_model(
        backbone = backbone,
        pretrained_params = pretrained_params,
        freeze_backbone = freeze_backbone,
        num_classes = num_classes,
    ).to(DEVICE)
    
    # MODE
    match mode:
        case "train": # - train -------
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

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
            )
            print(f"Loading best checkpoint '{best_ckpt}' and testing model")
            model.load_state_dict(torch.load(best_ckpt, map_location="cpu"))
            test(
                model,
                test_loader,
            )
        
        case "test":  # - test ---------
            test(
                model,
                test_loader,
            )
        
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



# ========================
# region CLI
# ========================


def get_checkpoints(root: Path|str) -> list[Path]:
    root = Path(root)
    return [ root / p.relative_to(root) for p in root.rglob("*.pt") ]

DEVICE: torch.device

LOSS_FUNCS = {
    "bce": nn.BCEWithLogitsLoss(),
    "focal": FocalLoss,
    "class_balanced": ClassBalancedLoss,
}

# TODO: implement
ATTENTION_MECHANISMS = {
    "SE": "Squeeze-and-Excitation: ...",
    "MHA": "Multi-head Attention: ...",
}

PRETRAINED_BACKBONES = {
    'resnet18':     './pretrained_backbone/ckpt_resnet18_ep50.pt',
    'effnet':       './pretrained_backbone/ckpt_efficientnet_ep50.pt',
}

if __name__ == "__main__":

    # ARGS
    # -------------------------------
    import argparse
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument("mode", nargs="?", default="train", choices=["train", "test", "predict", "none"])
    parser.add_argument('--dataset_path', default="./ODIR_dataset", help='Path to dataset root')
    parser.add_argument('--backbone', '-b', default="resnet18",
                            help='Which model to use as backbone (else: detects ackbone from --load_checkpoint)')
    parser.add_argument('--no_pretrained_params', '-npp', action='store_true', 
                            help="Don't load any params (re-initialize weights, train from scratch)")
    parser.add_argument('--load_checkpoint', '-ckp',
                            help="Path to checkpoint to load (relative to `checkpoints/`) (else: load pretrained backbone from `pretrained_backbone/`)")
    # parser.add_argument('--load_pretrained', '-pre', action='store_true', help="Use pretrained weights for model (in pretrained_backbone)")
    
    # train args
    parser.add_argument('--save_name', '-sn', help="Path to save best checkpoint (in checkpoints/)")
    parser.add_argument('--ft_mode', default="classifier", choices=["classifier", "all"],
                        help="Fine-tuning mode: which params to unfreeze")
    parser.add_argument('--loss_fn', default="bce", help="Loss function to use during training")
    parser.add_argument('--attention', help="Attention mechanism to use (use help to list options)")
    
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=32)

    # misc
    parser.add_argument('--predict_csv', default="onsite_test_submission", help="Path to csv to save output in predict mode")
    parser.add_argument('--checkpoints_dir', default="checkpoints", help="Folder to save and load checkpoints (default `checkpoints/`)")
    parser.add_argument('--list_checkpoints', action="store_true", help="List all detected checkpoints")

    parser.add_argument('--num_classes', type=int, default=3,
                            help="Number of classes we want to detect (changes shape of classifier)")
    
    args = parser.parse_args()
    
    
    # HANDLE ARGS
    # -------------------------------
    
    # backbone & checkpoint
    checkpoints = get_checkpoints(args.checkpoints_dir)
    if args.list_checkpoints:
        print(f"Checkpoints detected in '{args.checkpoints_dir}':")
        for i, f in enumerate(checkpoints):
            print(f"  {i+1:>3}: {f}")
        sys.exit()
    
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
        print('[PARAMS] using  checkpoint:', params_file)
        
        # autodetect backbone
        for bb in PRETRAINED_BACKBONES.keys():
            if bb.lower() in params_file:
                print(f"Detected backbone from checkpoint filename (overriding --backbone)")
                args.backbone = bb
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
    

    # save name
    savename = args.save_name
    if savename is None:
        savename = f"{args.backbone}_ft-{args.ft_mode}_lr-{args.lr}_{args.epochs}ep_best.pt"
    else:
        if not savename.endswith(".pt"):
            savename += ".pt"
    
    if os.path.exists( os.path.join(args.checkpoints_dir, savename) ) and args.mode == "train":
        if input(f"\033[31mImportant:\033[0m Checkpoint file ('{savename}') already exists. Are you sure you want to replace it?\n ('y' or 'yes') > "
                ).lower() not in ["y", "yes", "yeahboii"]:
            print(" ..quitting\n")
            sys.exit(0)
    
    # loss fn
    if args.loss_fn not in LOSS_FUNCS:
        if args.loss_fn != "help":
            print("[ERROR] No such loss function:", args.loss_fn)
        print("Available loss functions:")
        for i, (k, v) in enumerate(LOSS_FUNCS.items()):
            print(f"  {i+1:>3}: {k:<18} ({v})")
        sys.exit(2)
    
    # attention
    if args.attention is not None and args.attention not in ATTENTION_MECHANISMS:
        if args.attention != "help":
            print("[ERROR] No such attention mechanism:", args.attention)
        print("Available attention mechanisms:")
        for i, (k, v) in enumerate(ATTENTION_MECHANISMS.items()):
            print(f"  {i+1:>3}: {k:<18} ({v})")
        sys.exit(2)
    
    
    # MAIN
    # -------------------------------
    
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
        predict_csv = args.predict_csv,
        epochs = args.epochs,
        lr = args.lr, # default: 1e-5
        batch_size = args.batch_size, # default: 32
        num_classes = args.num_classes,
    )

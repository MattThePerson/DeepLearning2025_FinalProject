"""
Code for training and fine-tuning. By Matt Stirling. 
"""
import os
import sys
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

    train_loader =        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   =        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  =        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    onsite_test_loader  = DataLoader(onsite_test_ds, batch_size=1, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, onsite_test_loader




# ========================
# region BUILD MODEL
# ========================

def build_model(backbone="resnet18", num_classes=3):

    if backbone == "resnet18":
        model = models.resnet18(weights=None) #"IMAGENET1K_V1") # TODO: should this be None??
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "efficientnet":
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
        state_dict = torch.load(pretrained_params, map_location="cpu")
        try:
            model.load_state_dict(state_dict)
        except:
            print(f"ERROR: Incompatible backbone ({backbone}) and params file ({pretrained_params})\nexiting ...")
            sys.exit(2)
    
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
        for imgs, labels in tqdm(loader, colour="magenta"):
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
        save_name="checkpoints/best.pt",
    ):

    ensure_parent_exists(save_name)
    print('loss function:', loss_fn)
    
    # iterates
    best_val_loss = float("inf")
    for epoch in range(epochs):
        # print('training ...')
        model.train()
        train_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}", colour="purple"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset) # type: ignore

        # validation
        # print('evaluating ...')
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset) # type: ignore

        print(f"Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"    ..saving best checkpoint to: {save_name}")
            torch.save(model.state_dict(), save_name)

    return save_name


# ========================
# region MAIN
# ========================
def main(
        mode = "train",
        backbone = "resnet18",
        dataset_path = "dataset",
        pretrained_params: str|None = None,
        save_name: str|None = None,
        freeze_backbone = True, # freeze non-linear layers
        loss_fn = nn.BCEWithLogitsLoss,
        attention = None,
        predict_csv = "onsite_test_submission.csv",
        # save_dir="checkpoints",
        epochs=10, batch_size=32, lr=1e-4, img_size=256,
    ):
    
    train_loader, val_loader, test_loader, onsite_test_loader = get_dataloaders(dataset_path, img_size, batch_size)

    print('Building model')
    model = get_model(
        backbone = backbone,
        pretrained_params = pretrained_params,
        freeze_backbone = freeze_backbone,
    ).to(DEVICE)

    
    save_name = "checkpoints/test.pt"

    # MODE
    match mode:
        case "train": # - train -------
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

            print(f'Training {backbone} for {epochs} epochs')
            train(
                model,
                train_loader,
                val_loader,
                optimizer,
                loss_fn,
                epochs=epochs,
                save_name=save_name,
            )
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

LOSS_FUNCS = {
    "bce": nn.BCEWithLogitsLoss(),
    "focal": FocalLoss,
    "class_balanced": ClassBalancedLoss,
}

DEVICE: torch.device


if __name__ == "__main__":

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using device:', DEVICE)

    # ARGS
    # -------------------------------
    import argparse
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument("mode", nargs="?", default="train", choices=["train", "test", "predict", "none"])
    parser.add_argument('--dataset_path', default="./ODIR_dataset", help='Path to dataset root')
    parser.add_argument('--save_name', help="Name to give checkpoint file")

    # 
    parser.add_argument('--backbone', '-b', default="resnet18", help='Which model to use as backbone', choices=["resnet18", "efficientnet"])
    parser.add_argument('--use_pretrained', '-pre', action='store_true', help="Use pretrained weights for model (in pretrained_backbone)")
    parser.add_argument('--load_checkpoint', '-ckp', help="Path to fine-tuned params (checkpoint)")
    
    parser.add_argument('--ft_mode', default="classifier", choices=["classifier", "all"],
                        help="Fine-tuning mode: which params to unfreeze")

    # train args
    parser.add_argument('--loss_fn', default="bce", choices=["bce", "focal", "class_balanced"],
                        help="Loss function to use during training")
    parser.add_argument('--attention', choices=["SE", "MHA"],
                        help="Attention mechanism to use (Squeeze-and-Excitation or Multi-head Attention)")
    
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--predict_csv', default="onsite_test_submission", help="name of csv for predictions (can include folders)")
    
    args = parser.parse_args()
    
    
    # HANDLE ARGS
    # -------------------------------
    
    # params file
    params_file = None
    if args.use_pretrained:
        params_file = {
            'resnet18':     './pretrained_backbone/ckpt_resnet18_ep50.pt',
            'efficientnet': './pretrained_backbone/ckpt_efficientnet_ep50.pt',
        }[args.backbone]
        print('using pretrained backbone:', params_file)
    
    elif args.load_checkpoint:
        files = [ f for f in os.listdir('.') if f.endswith('.pt') ]
        folder = 'checkpoints'
        files.extend([ os.path.join(folder, f) for f in os.listdir(folder) ])
        params_file = next((x for x in files if args.load_checkpoint in x), None)
        if params_file is None:
            raise FileNotFoundError(f"No such checkpoint found: {args.load_checkpoint}")
        print('Using fine-tuned checkpoint:', params_file)

    else:
        print('No params file given, training model from scratch & enabling full fine-tuning mode')
        args.ft_mode = "all"

    # save name
    save_name = args.save_name # TODO: fix!!
    
    
    # MAIN
    # -------------------------------
    main(
        mode = args.mode,
        dataset_path = args.dataset_path,
        backbone = args.backbone,
        pretrained_params = params_file,
        save_name = save_name,
        freeze_backbone = (args.ft_mode == "classifier"),
        loss_fn = LOSS_FUNCS[args.loss_fn],
        predict_csv = args.predict_csv,
        epochs = args.epochs,
        lr = args.lr, # default: 1e-5
        batch_size = args.batch_size, # default: 32
    )

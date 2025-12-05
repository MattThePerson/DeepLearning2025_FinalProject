"""
Code for training and fine-tuning. By Matt Stirling. 
"""
import os
import pandas as pd
from PIL import Image
import numpy as  np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

# ========================
# RetinaMultiLabelDataset
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


# ========================
# region HELPERS
# ========================

def build_model(backbone="resnet18", num_classes=3, pretrained=True):

    if backbone == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1") # TODO: should this be None??
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "efficientnet":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        layer_fc: nn.Linear = model.classifier[1] # type: ignore[assignment]
        model.classifier[1] = nn.Linear(layer_fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    
    for p in model.parameters():
        p.requires_grad = True
    
    return model


def get_dataloaders(img_size=256, batch_size=32):
    
    # transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # dataset & dataloader
    train_ds = RetinaMultiLabelDataset(train_csv, train_image_dir, transform)
    val_ds   = RetinaMultiLabelDataset(val_csv, val_image_dir, transform)
    test_ds  = RetinaMultiLabelDataset(test_csv, test_image_dir, transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader


def freeze_non_linear_layers(model):
    """
    AKA: Freeze backbone and leave classifier (linear layers) unfrozen. 
    """
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze only Linear layers
    for m in model.modules():
        if isinstance(m, nn.Linear):
            for p in m.parameters():
                p.requires_grad = True
    return model


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
def predict(model, ):
    ...

# ========================
# region test
# ========================
def test(model, test_loader, device=torch.device("cpu"), save_dir="checkpoints", name="noname"):

    ckpt_path = os.path.join(save_dir, f"best_{name}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No such checkpoint: {ckpt_path}")
    
    print('loading checkpoint:', ckpt_path)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # evaluate
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    y_true = np.array(y_true) #torch.tensor(y_true).numpy()
    y_pred = np.array(y_pred) #torch.tensor(y_pred).numpy()

    # results to DataFrame
    disease_names = ["DR", "Glaucoma", "AMD"]
    results_data = []
    
    for i, disease in enumerate(disease_names):  # compute metrics for every disease
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        acc =       accuracy_score(y_t, y_p)
        precision = precision_score(y_t, y_p, average="macro", zero_division=0)
        recall =    recall_score(y_t, y_p, average="macro", zero_division=0)
        f1 =        f1_score(y_t, y_p, average="macro", zero_division=0)
        kappa =     cohen_kappa_score(y_t, y_p)

        results_data.append([disease, acc, precision, recall, f1, kappa])

    results = pd.DataFrame(data=results_data, columns=["Disease", "Accuracy", "Precision", "Recall", "F1-score", "Kappa"])
    print(results)
    
    return results


# ========================
# region train
# ========================
def train(model, epochs, train_loader, val_loader, optimizer, criterion, device=torch.device("cpu"), save_dir='checkpoints', name="noname"):

    # training
    best_val_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"best_{name}.pt")
    
    # run epochs
    for epoch in range(epochs):
        print('training ...')
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset) # type: ignore

        # validation
        print('evaluating ...')
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset) # type: ignore

        print(f"epoch {epoch+1}/{epochs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"    ...saved best model {ckpt_path}")



# ========================
# region MAIN
# ========================
def main(backbone, train_csv, val_csv, test_csv, train_image_dir, val_image_dir, test_image_dir, 
            loss_fn,
            mode="train", run_name="noname",
            epochs=10, batch_size=32, lr=1e-4, img_size=256, save_dir="checkpoints", pretrained_backbone=None,
    ):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using device:', device)

    # data loaders
    print('creating data loaders')
    train_loader, val_loader, test_loader = get_dataloaders(img_size, batch_size)

    # model
    print('building model with backbone')
    model = build_model(backbone, num_classes=3, pretrained=False).to(device)

    # loss & optimizer
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # load pretrained backbone
    if pretrained_backbone is not None:
        print('loading pretrained backbone:', pretrained_backbone)
        state_dict = torch.load(pretrained_backbone, map_location="cpu")
        model.load_state_dict(state_dict)
    
    # TRAIN
    if mode == "train":
        print(f'Training {backbone} for {epochs} epochs')
        train(model, epochs, train_loader, val_loader, optimizer, loss_fn, device,
            save_dir=save_dir, name=backbone)
    
    # TEST
    print(f'Testing ...')
    test(model, test_loader, device,
         save_dir=save_dir, name=backbone)
    
    # PREDICT (csv)



# ========================
# region CLI
# ========================

LOSS_FUNCS = {
    "bce": nn.BCEWithLogitsLoss,
    # "focal": FocalLoss,
    # "class_balanced": ClassBalancedLoss,
}

if __name__ == "__main__":

    # args ------------------
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone', default="resnet18", help='Backbone for fine-tuned model', choices=["resnet18", "efficientnet"])
    parser.add_argument('--scratch', action='store_true', help="Do not load pretrained params")
    parser.add_argument('--full-ft', action='store_true', help="Train whole network. Else: Train only classifier (backbone forzen)")

    # 
    parser.add_argument('--mode', default='train', choices=["train", "test", "predict"], help="")
    parser.add_argument('--run-name', help="")

    # train args
    parser.add_argument('--loss_fn', default="bce", choices=["bce", "focal", "class_balanced"], 
                        help="Loss function to use during training")
    parser.add_argument('--attention', choices=["SE", "MHA"], 
                        help="Attention mechanism to use (Squeeze-and-Excitation or Multi-head Attention)")
    
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=32)
    
    args = parser.parse_args()
    
    if args.scratch:
        print('training from scratch, enabling full fine tuning')
        args.full_ft = True
    
    # paths -----------------
    DATASET_PATH = "./ODIR_dataset"
    
    train_csv = os.path.join( DATASET_PATH, "labels/train.csv" )
    val_csv   = os.path.join( DATASET_PATH, "labels/val.csv" )
    test_csv  = os.path.join( DATASET_PATH, "labels/offsite_test.csv" )
    
    train_image_dir = os.path.join( DATASET_PATH, "images/train" )
    val_image_dir =   os.path.join( DATASET_PATH, "images/val" )
    test_image_dir =  os.path.join( DATASET_PATH, "images/offsite_test" )

    # pretrained backbone ----
    pretrained_backbones = {
        'resnet18': './pretrained_backbone/ckpt_resnet18_ep50.pt',
        'efficientnet': './pretrained_backbone/ckpt_efficientnet_ep50.pt',
    }
    pretrained_backbone = None
    if not args.from_scratch:
        pretrained_backbone = pretrained_backbones[args.backbone]
    
    # run -------------------
    main(args.backbone, train_csv, val_csv, test_csv, train_image_dir, val_image_dir, test_image_dir,
            epochs=args.epochs,
            mode=args.mode,
            
            lr=args.lr, # default: 1e-5
            batch_size=args.batch_size, # default: 32
            pretrained_backbone=pretrained_backbone,

            loss_fn=LOSS_FUNCS[args.loss_fn],

            img_size=256,
    )

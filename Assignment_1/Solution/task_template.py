import os
import sys
import torch
import pandas as pd
import requests
import random
import argparse

from pathlib import Path
from torch.utils.data import Dataset
from torchvision.models import resnet18
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from xgboost import XGBClassifier

# # config
BASE = Path(__file__).parent
PUB_PATH = BASE / "pub.pt"
PRIV_PATH = BASE / "priv.pt"
MODEL_PATH = BASE / "model.pt"
OUTPUT_CSV = BASE / "submission.csv"

BASE_URL = "http://34.63.153.158"   #DONOT CHANGE
API_KEY = "7e1efa852453a942fc9e67e35c5d2377"
TASK_ID = "01-mia"  #DONOT CHANGE

#added
NUM_SHADOW_MODELS = 80   
SHADOW_EPOCHS = 20        
SHADOW_TRAIN_RATIO = 0.5  
BATCH_SIZE = 64
LR = 0.001



# dataset classes
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index):
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]



def get_balanced_shadow_indices(dataset, split_ratio=0.5):
    """
    Ensures equal number of samples per class in the shadow training set.
    e.g. 900 samples, 9 classes → 100 per class, 50 IN / 50 OUT per class
    """
    # Group all indices by class
    class_member_indices = defaultdict(lambda: defaultdict(list))
    for idx in range(len(dataset)):
        _, _, label, membership = dataset[idx]
        # label = label.item() if isinstance(label, torch.Tensor) else label
        class_member_indices[label][membership].append(idx)

    # Find the minimum class size to ensure equal representation
    min_count = min(
        len(class_member_indices[cls][mem])
        for cls in class_member_indices
        for mem in [0, 1]
        if mem in class_member_indices[cls]
    )
    n_per_class = int(np.floor(split_ratio * min_count))

    print(f"  Min count per (class, membership) group: {min_count}", flush=True)
    print(f"  Samples per group selected as IN: {n_per_class}", flush=True)

    shadow_indices = []

    for class_label in sorted(class_member_indices.keys()):
        for membership_val in [0, 1]:
            indices = class_member_indices[class_label][membership_val]

            # Equalize to min_count, then take n_per_group as IN
            equalized   = np.random.choice(indices, size=min_count, replace=False)
            selected_in = equalized[:n_per_class].tolist()
            shadow_indices.extend(selected_in)

            print(f"  Class {class_label} | membership={membership_val}: "
                  f"{len(indices)} total → {n_per_class} IN selected", flush=True)
    return shadow_indices



# load datasets
print("Loading datasets...", flush=True)
pub_ds = torch.load(PUB_PATH, weights_only=False)
priv_ds = torch.load(PRIV_PATH, weights_only=False)
priv_ds.membership = [-1] * len(priv_ds) # dummy membership to remove none error
scaler = StandardScaler()


# normalization (same as training)
MEAN = [0.7406, 0.5331, 0.7059]
STD = [0.1491, 0.1864, 0.1301]

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Normalize(mean=MEAN, std=STD),
])



pub_ds.transform = transform    # attach normalization after loading
priv_ds.transform = transform





# load model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader_pub = DataLoader(pub_ds, batch_size=BATCH_SIZE, shuffle=False)
criterion_train = torch.nn.CrossEntropyLoss()
criterion_attack = torch.nn.CrossEntropyLoss(reduction="none")


def get_model_architecture():
    model = resnet18(weights=None)
    model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(512, 9)
    return model.to(device)

attack_X = []
attack_y = []


#split pub_ds into In/Out for shadow training with equal ratio
dataset_size = len(pub_ds)
indices = list(range(dataset_size))
split = int(np.floor(SHADOW_TRAIN_RATIO * dataset_size))
print(f"\n--- Training {NUM_SHADOW_MODELS} Shadow Models ---", flush=True)
attack_X_per_class = defaultdict(lambda: {'in': [], 'out': []})

# print(f"Shadow training: {split} In, {dataset_size - np.split} Out", flush=True)
for shadow_idx in range(NUM_SHADOW_MODELS):
    # shadow_indices = get_balanced_shadow_indices(pub_ds, split_ratio=SHADOW_TRAIN_RATIO)
    # if shadow model already exists, skip training and directly compute attack features
    shadow_model_path = BASE / f"shadow_model_{shadow_idx}.pth"
    # if shadow_model_path.exists():
    #     print(f"Loading existing shadow model {shadow_idx}...", flush=True)
    #     shadow_model = get_model_architecture()
    #     shadow_model.load_state_dict(torch.load(shadow_model_path))
    #     # shadow_model.eval()
    # else:
    np.random.shuffle(indices)
    shadow_indices = indices[:split]
    in_set = set(shadow_indices)

    shadow_subset = torch.utils.data.Subset(pub_ds, shadow_indices)


    shadow_train_loader = DataLoader(shadow_subset, batch_size=BATCH_SIZE, shuffle=True)

    shadow_model = get_model_architecture()
    optimizer = torch.optim.Adam(shadow_model.parameters(), lr=LR)


    shadow_model.train()
    for epoch in range(SHADOW_EPOCHS):
        total_loss = 0
        correct = 0
        for _, imgs, labels, _ in shadow_train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = shadow_model(imgs)
            loss = criterion_train(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(imgs)
            correct += (logits.argmax(dim=1) == labels).sum().item()
        accuracy = correct / len(shadow_train_loader.dataset)

        avg_loss = total_loss / len(shadow_train_loader.dataset)
        print(f"Shadow Model {shadow_idx+1}/{NUM_SHADOW_MODELS}, Epoch {epoch+1}/{SHADOW_EPOCHS}, Loss: {avg_loss:.4f}, Correct: {correct}, Accuracy: {accuracy:.4f}", flush=True)
        #save the model
        # torch.save(shadow_model.state_dict(), f"shadow_model_{shadow_idx}.pth")
    shadow_model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader_pub):
            id_, imgs, labels, membership = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = shadow_model(imgs)
            probs = F.softmax(logits, dim=1)
            sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
            correct_prob = probs.gather(1, labels.view(-1,1)).squeeze(1)
            loss = criterion_attack(logits, labels)
            # print(f"sorted_probs shape: {sorted_probs.shape}")
            # print(f"sored_probs sample: {sorted_probs[0]}")
            # print(f"correct_prob shape: {correct_prob.shape}")
            # print(f"sorted_probs shape: {sorted_probs.shape}")
            # print(f"loss shape: {loss.shape}")
            features = torch.cat([
                # loss.unsqueeze(1),  # 1 feature: loss
                correct_prob.unsqueeze(1),  # 1 feature: correct class probability
                sorted_probs
            ], dim=1).cpu().numpy()  # total 1 + 1 + 9 = 11 features
            
            global_indices = range(batch_idx * BATCH_SIZE, batch_idx * BATCH_SIZE + len(id_))
            batch_shadow_labels = [1 if i in in_set else 0 for i in global_indices]
            attack_X.extend(features)
            attack_y.extend(batch_shadow_labels)
            # for feat, lbl, class_idx in zip(features, batch_shadow_labels, labels.cpu().numpy()):
            #     if lbl == 1:
            #         attack_X_per_class[class_idx]['in'].append(feat)
            #     else:
            #         attack_X_per_class[class_idx]['out'].append(feat)
            
# min_count = min(
#     min(len(v['in']), len(v['out']))
#     for v in attack_X_per_class.values()
# )
# print(f"Min count per class (IN or OUT): {min_count}", flush=True)
# count_per_class = {cls: {'in': len(v['in']), 'out': len(v['out'])} for cls, v in attack_X_per_class.items()}
# print(f"Attack X per class distribution before balancing: {count_per_class}", flush=True)


# for class_idx in sorted(attack_X_per_class.keys()):
#     in_samples  = np.array(attack_X_per_class[class_idx]['in'])
#     out_samples = np.array(attack_X_per_class[class_idx]['out'])

#     # Subsample to min_count for balance
#     in_idx  = np.random.choice(len(in_samples),  min_count, replace=False)
#     out_idx = np.random.choice(len(out_samples), min_count, replace=False)

#     attack_X.extend(in_samples[in_idx])
#     attack_y.extend([1] * min_count)
#     attack_X.extend(out_samples[out_idx])
#     attack_y.extend([0] * min_count)

#     print(f"Class {class_idx}: {min_count} IN + {min_count} OUT = {min_count*2} total", flush=True)

# attack_X = np.array(attack_X)
# attack_y = np.array(attack_y)
# print(f"\nFinal attack dataset: {attack_X.shape}, {attack_y.shape}", flush=True)
attack_X = np.array(attack_X)
attack_y = np.array(attack_y)
print("\n--- Attack Dataset ---", flush=True)
print(attack_X.shape, attack_y.shape, flush=True)
# attack_X = scaler.fit_transform(attack_X)
# randomize attack_x and attack_y together
indices = np.random.permutation(len(attack_X))
attack_X = attack_X[indices]
attack_y = attack_y[indices]
X_train, X_val, y_train, y_val = train_test_split(attack_X, attack_y, test_size=0.2)
print(f"Attack training set: {X_train.shape}, {y_train.shape}", flush=True)
print(f"Attack validation set: {X_val.shape}, {y_val.shape}", flush=True)
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.int32)
X_val = X_val.astype(np.float32)
y_val = y_val.astype(np.int32)

print("\n--- Training Attack Model ---", flush=True)

# attack_model = RandomForestClassifier(
#     n_estimators=200,      
#     max_depth=30,          
#     min_samples_leaf=50,   
#     max_features=None,     
#     n_jobs=-1,             
#     random_state=42,
#     verbose=1              
# )
# attack_model = RandomForestClassifier(
#     n_estimators=300,
#     max_depth=None,
#     min_samples_split=10,
#     min_samples_leaf=5,
#     max_features=None,
#     class_weight="balanced",
#     n_jobs=-1,
#     random_state=42
# )
# attack_model.fit(X_train, y_train)

# class AttackMLP(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.netwk = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.BatchNorm1d(128), 
#             nn.ReLU(),           
#             nn.Linear(128, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 2)
#         )
#     def forward(self, x):
#         return self.netwk(x)
# print("Initializing attack model, optimizer", flush=True)
# attack_model = AttackMLP(attack_X.shape[1]).to(device)

# optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.001, weight_decay=1e-7)
# criterion = nn.CrossEntropyLoss()
# ds_train = torch.utils.data.TensorDataset(X_train, y_train)
# loader_train = DataLoader(ds_train, batch_size=512, shuffle=True)
# loader_val = DataLoader(torch.utils.data.TensorDataset(X_val, y_val), batch_size=256)
# print("Starting attack model training loop", flush=True)
# for epoch in range(20):
#     attack_model.train()    
#     total_loss = 0
#     correct = 0
#     for batch_X, batch_y in loader_train:
#         optimizer.zero_grad()
#         logits = attack_model(batch_X)
#         loss = criterion(logits, batch_y)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * len(batch_X)
#         correct += (logits.argmax(dim=1) == batch_y).sum().item()
#     attack_model.eval()
#     val_loss = 0
#     val_correct = 0
#     all_preds = []
#     all_labels = []
#     with torch.no_grad():
#         for batch_X, batch_y in loader_val:
#             logits = attack_model(batch_X)
#             loss = criterion(logits, batch_y)
#             val_loss += loss.item() * len(batch_X)
#             val_correct += (logits.argmax(dim=1) == batch_y).sum().item()
#             probs = F.softmax(logits, dim=1)
#             all_preds.extend(probs.cpu().numpy())
#             all_labels.extend(batch_y.cpu().numpy())
#     avg_loss = val_loss / len(loader_val.dataset)
#     accuracy = val_correct / len(loader_val.dataset)
#     all_probs = np.array(all_preds)
#     all_labels = np.array(all_labels)
#     fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
#     val_auc      = auc(fpr, tpr)

#     # TPR @ 5% FPR
#     idx = max(0, min(np.searchsorted(fpr, 0.05, side='right') - 1, len(tpr) - 1))
#     tpr_at_5 = tpr[idx]

#     print(f"Epoch {epoch+1}/20 | "
#     #   f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
#       f"Val Loss: {avg_loss:.4f} Acc: {accuracy:.4f} | "
#       f"AUC: {val_auc:.4f} TPR@5%FPR: {tpr_at_5:.4f}", flush=True)
#     # print(f"Epoch {epoch+1}/50, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}", flush=True)

print("\n--- Training Attack Model (XGBoost) ---", flush=True)


attack_model = XGBClassifier(
    n_estimators=500,          
    max_depth=6,               
    learning_rate=0.05,        
    subsample=0.8,             
    colsample_bytree=0.8,      
    random_state=42,
    eval_metric="auc",         
    early_stopping_rounds=50,  
    tree_method="hist",        
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Fit the model
attack_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=50  # Print progress every 50 trees
)

# Evaluate the best iteration
val_probs = attack_model.predict_proba(X_val)[:, 1]
fpr, tpr, _ = roc_curve(y_val, val_probs) 
val_auc = auc(fpr, tpr)

idx = max(0, min(np.searchsorted(fpr, 0.05, side='right') - 1, len(tpr) - 1))
tpr_at_5 = tpr[idx]

print(f"\nXGBoost Final Validation AUC: {val_auc:.4f} | TPR@5%FPR: {tpr_at_5:.4f}", flush=True)




# rf_train_acc = attack_model.score(X_val, y_val)
# print(f"Attack model training accuracy: {rf_train_acc:.4f}", flush=True)
# fpr, tpr, _ = roc_curve(y_val, attack_model.predict_proba(X_val)[:, 1])
# idx = np.searchsorted(fpr, 0.05)
# print(f"TPR @ 5% FPR: {tpr[idx]:.4f}", flush=True)

print("Loading model...", flush=True)
model = resnet18(weights=None)
model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
model.maxpool = torch.nn.Identity()
model.fc = torch.nn.Linear(512, 9)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model = model.to(device)
model.eval()

priv_ids = []
priv_featrues = []
loader_priv = DataLoader(priv_ds, batch_size=BATCH_SIZE, shuffle=False)
with torch.no_grad():
    for batch_idx, batch in enumerate(loader_priv):
        # print elements in batch for debugging
        # print(batch_idx, [type(x) for x in batch], [x.shape if isinstance(x, torch.Tensor) else len(x) for x in batch])
        id, imgs, labels, _ = batch
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1)
        sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
        loss = criterion_attack(logits, labels)
        correct_prob = probs.gather(1, labels.view(-1,1)).squeeze(1)
        features = torch.cat([
            # loss.unsqueeze(1),  # 1 feature: loss
            correct_prob.unsqueeze(1),  # 1 feature: correct class probability
            sorted_probs
        ], dim=1).cpu().numpy()  # total 1 + 1 + 9 = 11 features
        priv_ids.extend(id.tolist())
        priv_featrues.extend(features)

print("\n--- Predicting Membership ---", flush=True)
priv_featrues = np.array(priv_featrues)
# priv_featrues_scaled = scaler.transform(priv_featrues)

membership_scores = attack_model.predict_proba(priv_featrues)[:, 1]
# with torch.no_grad():
#     membership_scores = F.softmax(
#         attack_model(torch.tensor(priv_featrues, dtype=torch.float32).to(device)),
#         dim=1
#     ).detach().cpu().numpy()[:, 1]



#save submission
df = pd.DataFrame({
        "id": priv_ids,
        "score": membership_scores
})

df.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV, flush=True)





# -------------------------








# create random submission (remove this later or it will rewrite your actual submission)
# print("Creating random submission...")
# ids = [str(i) for i in priv_ds.ids]

# df = pd.DataFrame({
#     "id": ids,
#     "score": [random.random() for _ in ids]
# })

# df.to_csv(OUTPUT_CSV, index=False)
# print("Saved:", OUTPUT_CSV)

###############

# all_scores = []
# all_labels = []
# with torch.no_grad():
#     for batch in loader_pub:
#         id_, imgs, labels, membership = batch
#         y_true = membership.cpu().numpy() if isinstance(membership, torch.Tensor) else np.array(membership)
#         imgs = imgs.to(device)
#         labels = labels.to(device)
#         logits = model(imgs)
#         loss = criterion(logits, labels).cpu().numpy()
#         scores = -loss      # negate: lower loss = more member-like

#         all_labels.extend(y_true.tolist())
#         all_scores.extend(scores.tolist())

# all_labels = np.array(all_labels)
# all_scores = np.array(all_scores)
# fpr, tpr, _ = roc_curve(all_labels, all_scores)
# roc_auc = auc(fpr, tpr)

# plt.figure()
# plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
# plt.plot([0,1], [0,1], linestyle="--", color="gray", label="chance")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC curve (membership scores)")
# plt.legend(loc="lower right")
# roc_path = BASE / "roc.png"
# plt.savefig(roc_path)
# plt.close()

# print(f"ROC AUC: {roc_auc:.6f}")
# print("Saved ROC plot to:", roc_path)



# all_ids = []
# all_scores = []
# criterion = torch.nn.CrossEntropyLoss(reduction="none")  # one loss value per sample

# with torch.no_grad():
#     for id_, imgs, labels, _ in loader:
#         imgs = imgs.to(device)
#         labels = labels.to(device)
#         logits = model(imgs)
#         loss = criterion(logits, labels).cpu().numpy()
#         scores = -loss      # negate: lower loss = more member-like

#         all_ids.extend(id_.tolist())
#         all_scores.extend(scores.tolist())

# df = pd.DataFrame({
#     "id": all_ids,
#     "score": all_scores
# })

# df.to_csv(OUTPUT_CSV, index=False)
# print("Saved:", OUTPUT_CSV)



# submit
def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)

parser = argparse.ArgumentParser(description="Submit a CSV file to the server.")
args = parser.parse_args()

submit_path = OUTPUT_CSV

if not submit_path.exists():
    die(f"File not found: {submit_path}")

try:
    with open(submit_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": API_KEY},
            files={"file": (submit_path.name, f, "application/csv")},
            timeout=(10, 600),
        )
    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}

    if resp.status_code == 413:
        die("Upload rejected: file too large (HTTP 413).")

    resp.raise_for_status()

    print("Successfully submitted.")
    print("Server response:", body)
    submission_id = body.get("submission_id")
    if submission_id:
        print(f"Submission ID: {submission_id}")

except requests.exceptions.RequestException as e:
    detail = getattr(e, "response", None)
    print(f"Submission error: {e}")
    if detail is not None:
        try:
            print("Server response:", detail.json())
        except Exception:
            print("Server response (text):", detail.text)
    sys.exit(1)
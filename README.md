# Membership Inference Attack: Experiment Log & Progression

This repository documents the iterative process and experimental progression of implementing a Membership Inference Attack (MIA). Below is a chronological breakdown of the strategies we tested, from basic thresholding to advanced shadow model techniques using deep learning and gradient boosting.

## Phase 1: The Baseline – Threshold-Based Attack

Initially, we wanted to see if a simple threshold-based attack would be sufficient. The idea was to check the loss/confidence distribution between the "in" (member) and "out" (non-member) datasets.

We plotted the ROC curve to analyze the Area Under the Curve (AUC). The resulting AUC was close to 0.5, indicating that the distributions were heavily overlapping. This confirmed that a simple threshold-based attack would not perform better than random chance.

<b>Implementation:</b>

```
all_scores = []
all_labels = []
with torch.no_grad():
    for batch in loader_pub:
        id_, imgs, labels, membership = batch
        y_true = membership.cpu().numpy() if isinstance(membership, torch.Tensor) else np.array(membership)
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels).cpu().numpy()
        scores = -loss      # negate: lower loss = more member-like

        all_labels.extend(y_true.tolist())
        all_scores.extend(scores.tolist())

all_labels = np.array(all_labels)
all_scores = np.array(all_scores)
fpr, tpr, _ = roc_curve(all_labels, all_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
plt.plot([0,1], [0,1], linestyle="--", color="gray", label="chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve (membership scores)")
plt.legend(loc="lower right")
roc_path = BASE / "roc.png"
plt.savefig(roc_path)
plt.close()

print(f"ROC AUC: {roc_auc:.6f}")
print("Saved ROC plot to:", roc_path)
```

## Phase 2: Shadow Model Attacks with Random Forest

Realizing thresholding wasn't viable, we moved to a Shadow Model architecture.

### Iteration 1: The 3-Model Setup

We started small, training just 3 shadow models using randomly selected 50% subsets of the public dataset. We then trained a basic Random Forest classifier on the attack data:

```
attack_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
attack_model.fit(attack_X, attack_y)
```

[Output Reference](https://github.com/bishaljoshi08/TML_MIA/blob/main/Assignment_1/output_logs/run.sh.mia.83861.out).

### Iteration 2: Scaling to 80 Models

To generate a more robust attack dataset, we scaled up to 80 shadow models (still using the 50% random subset strategy).
[Output Reference](https://github.com/bishaljoshi08/TML_MIA/blob/main/Assignment_1/output_logs/run.sh.mia.83862.out).

Scaling the data exposed a flaw: the simple Random Forest lacked the capacity to handle the increased data volume, resulting in poor training accuracy. We bumped up the model complexity to compensate:

```
attack_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=30,
    min_samples_leaf=50,
    max_features=None,
    n_jobs=-1,
    random_state=42,
    verbose=1
)
```

This adjustment successfully pushed our training accuracy above 90%.
[Output Reference](https://github.com/bishaljoshi08/TML_MIA/blob/main/Assignment_1/output_logs/run.sh.mia.83885.out).

## Phase 3: Transitioning to Neural Networks (MLPs)

Looking for better generalization, we replaced the Random Forest with a simple Neural Network.

### Feature Engineering & Initial NN

Our input features consisted of the sorted probabilities from the shadow models appended with the correct class probability (resulting in 10 input features total).

Initially, we used a lightweight NN (Input -> 64 -> 32 -> 2 with tanh activations). To manage processing times, we heavily capped the data, using only 10% of the data from each class (using the class with the lowest datapoint count as our absolute limit).

[Output Reference](https://github.com/bishaljoshi08/TML_MIA/blob/main/Assignment_1/output_logs/run.sh.mia.83966.out).

### Class Balancing Attempts

To enforce a stricter class distribution, we tried balancing everything perfectly: we took the smallest class member count in the attack dataset and applied that exact limit across all other classes and membership pairs.

[Output Reference](https://github.com/bishaljoshi08/TML_MIA/blob/main/Assignment_1/output_logs/run.sh.mia.83986.out).

### Deeper Architecture

We eventually increased our data usage back to 50% per shadow model (while keeping the minimum-class baseline logic). To handle this, we upgraded to a deeper MLP architecture with Batch Normalization and ReLU activations:

```
class AttackMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.netwk = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.netwk(x)
```

[Output Reference](https://github.com/bishaljoshi08/TML_MIA/blob/main/Assignment_1/output_logs/run.sh.mia.83991.out).

### Learning Rate Tuning:

- We initially tried a very low learning rate (`0.00001`) with a large batch size (`512`), assuming it would stabilize training. It didn't perform well.
- We then bumped the learning rate to `0.001`, which yielded a slight improvement on our custom validation set.

[Output Reference](https://github.com/bishaljoshi08/TML_MIA/blob/main/Assignment_1/output_logs/run.sh.mia.84019.out).

## Phase 4: Removing Constraints & Embracing XGBoost

Despite the architecture tweaks, our validation metrics were plateauing. After consulting with an LLM for strategic advice, we realized a critical flaw: <b>artificial data capping was destroying our natural data distribution.</b>

Because the target model was trained on a specific distribution, our shadow models and attack models needed to experience that exact same distribution to be effective.

# The Final Approach

1. <b>Removed all data capping</b> when training both the shadow models and the attack classifier.
2. Switched out the MLP for an <b>XGBoost Classifier</b>, which handles tabular probability data exceptionally well.

```
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
```

<b>Results:</b> This final uncapped XGBoost approach yielded our best results, achieving a True Positive Rate (TPR) of <b>0.07 at a 5% False Positive Rate (FPR)</b> on the validation set.
[Output Reference](https://github.com/bishaljoshi08/TML_MIA/blob/d2da5c2c3da6b5704dbdee2051eeb567ba343d20/Assignment_1/output_logs/run.sh.mia.84204.out#L1632).

## Recreating the final approach:

Use this [submission file](https://github.com/bishaljoshi08/TML_MIA/blob/main/Assignment_1/Submission/mia.sub) to submit the `task_template.py` to the cluster

### Note:

Due to the use of high-performance computing cluster, we modified the .py file itself. While the corresponding .py files in this submission contain commented-out code blocks reflecting these changes, the exact iterations of the scripts used for those runs were not preserved as separate files. However, all relevant experimental results and logs are available in the provided output files. Moving forward, we will implement a more rigorous version control process—either by maintaining distinct scripts for cluster environments or documenting all iterations within a unified notebook—to ensure full reproducibility of the codebase alongside the results.

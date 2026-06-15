"""
==============================================================================
 Output Range Analysis for Deep Feedforward Neural Networks
 Dataset: Breast Cancer Wisconsin (Kaggle)
==============================================================================
"""

# ============================ STEP 1 - Import Libraries ======================
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                 # headless: save figures instead of showing
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, accuracy_score, f1_score, precision_score,
                             recall_score)

import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)                    # reproducibility

os.makedirs("figures", exist_ok=True)
print("STEP 1  Libraries loaded successfully.")

# ============================ STEP 2 - Load Dataset ==========================
CSV = os.environ.get("CSV", "breast_cancer.csv")
df = pd.read_csv(CSV)
print("\nSTEP 2  Dataset loaded:", df.shape)
print(df.head())

# ============================ STEP 3 - Clean Data ============================
df = df.drop(["id", "Unnamed: 32"], axis=1, errors="ignore")     # drop useless cols
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})          # Malignant=1, Benign=0
print("\nSTEP 3  After cleaning:", df.shape)

# ============================ STEP 4 - Simulate Partially Labeled Data ========
# Real-world labels are expensive; we mask ~20% to show awareness of incomplete
# supervision. The supervised network still trains on the FULLY labeled df.
df_partial = df.copy()
missing_percentage = 0.20
num_missing = int(len(df_partial) * missing_percentage)
missing_indices = np.random.choice(df_partial.index, num_missing, replace=False)
df_partial.loc[missing_indices, "diagnosis"] = np.nan
print(f"\nSTEP 4  Simulated {missing_percentage*100:.0f}% missing labels "
      f"(~{num_missing} samples). Continuing with the fully labeled data.")

# ============================ STEP 5 - Class Imbalance Analysis ===============
counts = df["diagnosis"].value_counts()
print(f"\nSTEP 5  Benign (0): {counts[0]} | Malignant (1): {counts[1]} | "
      f"Imbalance ratio: {counts[0]/counts[1]:.2f}x")

plt.figure(figsize=(5, 4))
plt.bar(["Benign (0)", "Malignant (1)"], counts.values,
        color=["steelblue", "tomato"], edgecolor="black")
plt.title("Class Distribution - Breast Cancer Dataset"); plt.ylabel("Count")
for i, v in enumerate(counts.values):
    plt.text(i, v + 3, str(v), ha="center", fontweight="bold")
plt.tight_layout(); plt.savefig("figures/01_class_distribution.png", dpi=120); plt.close()

# Class weights (computed for diagnostic insight, as in the notebook)
n_total      = len(df)
weight_benign    = n_total / (2 * counts[0])
weight_malignant = n_total / (2 * counts[1])
print(f"        Class weights -> Benign: {weight_benign:.3f}, Malignant: {weight_malignant:.3f}")

# ============================ STEP 6 - Noise Injection ========================
# Inject small Gaussian noise (2% of each feature's std) to test robustness -
# directly linked to the paper's bounded input perturbations.
X_orig = df.drop("diagnosis", axis=1)
y      = df["diagnosis"]

X_noisy = X_orig.copy()
noise_cols = ["radius_mean", "texture_mean", "area_mean", "smoothness_mean", "compactness_mean"]
for col in noise_cols:
    noise = np.random.normal(0, 0.02 * X_orig[col].std(), size=len(X_orig))
    X_noisy[col] = X_orig[col] + noise

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("Noise Injection - Original vs Noisy Distribution", fontweight="bold")
for i, col in enumerate(["radius_mean", "area_mean"]):
    axes[i].hist(X_orig[col],  bins=30, alpha=0.6, label="Original", color="steelblue")
    axes[i].hist(X_noisy[col], bins=30, alpha=0.6, label="Noisy",    color="tomato")
    axes[i].set_title(col); axes[i].legend(); axes[i].set_ylabel("Frequency")
plt.tight_layout(); plt.savefig("figures/02_noise_injection.png", dpi=120); plt.close()
print("\nSTEP 6  Noise injected on:", noise_cols)

X = X_noisy.copy()        # use noisy data going forward (more realistic)

# ============================ STEP 7 - Feature Scaling & PCA ==================
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"\nSTEP 7  Scaled features: {X_scaled.shape[1]} dimensions")

pca_full = PCA().fit(X_scaled)
cum_var  = np.cumsum(pca_full.explained_variance_ratio_)
n_for_95 = int(np.searchsorted(cum_var, 0.95) + 1)
print(f"        {n_for_95} components explain 95% of variance "
      f"({X_scaled.shape[1]} -> {n_for_95})")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("PCA - High Dimensionality Analysis", fontweight="bold")
axes[0].bar(range(1, 31), pca_full.explained_variance_ratio_ * 100,
            color="steelblue", edgecolor="black", linewidth=0.4)
axes[0].set_xlabel("Principal Component"); axes[0].set_ylabel("Explained Variance (%)")
axes[0].set_title("Scree Plot")
axes[1].plot(range(1, 31), cum_var * 100, "o-", color="purple", linewidth=2, markersize=5)
axes[1].axhline(95, color="gray", linestyle="--", label="95% threshold")
axes[1].axvline(n_for_95, color="orange", linestyle="--", label=f"{n_for_95} components")
axes[1].set_xlabel("Number of Components"); axes[1].set_ylabel("Cumulative Variance (%)")
axes[1].set_title("Cumulative Variance"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("figures/03_pca_analysis.png", dpi=120); plt.close()

pca2   = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X_scaled)
plt.figure(figsize=(6, 5))
for label, color, name in [(0, "steelblue", "Benign"), (1, "tomato", "Malignant")]:
    mask = y.values == label
    plt.scatter(X_pca2[mask, 0], X_pca2[mask, 1], c=color, label=name,
                alpha=0.6, edgecolors="k", linewidths=0.2, s=30)
plt.xlabel(f"PC1 ({pca_full.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca_full.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("2D PCA Projection - Class Separability"); plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("figures/04_pca_scatter.png", dpi=120); plt.close()

# ============================ STEP 8 - Train/Test Split ======================
# Keep an index array so we can recover RAW (unscaled) rows for the website export.
idx = np.arange(len(X))
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_scaled, y, idx, test_size=0.2, random_state=42, stratify=y
)
print(f"\nSTEP 8  Train: {X_train.shape} | Test: {X_test.shape}")

# ============================ STEP 9 - Deep Feedforward NN (ReLU) =============
# Architecture: input(30) -> Dense(64,ReLU) -> Dense(32,ReLU) -> Dense(1,Sigmoid)
class NeuralNet:
    def __init__(self, input_size):
        # He (Kaiming) initialization - best practice for ReLU networks
        self.W1 = np.random.randn(input_size, 64) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, 64))
        self.W2 = np.random.randn(64, 32) * np.sqrt(2.0 / 64)
        self.b2 = np.zeros((1, 32))
        self.W3 = np.random.randn(32, 1) * np.sqrt(2.0 / 32)
        self.b3 = np.zeros((1, 1))

    def relu(self, z):       return np.maximum(0, z)
    def relu_grad(self, z):  return (z > 0).astype(float)
    def sigmoid(self, z):    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1; self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2; self.a2 = self.relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3; self.out = self.sigmoid(self.z3)
        return self.out

    def train(self, X, y, lr=0.005, epochs=1000):
        y = y.reshape(-1, 1)
        loss_history = []
        for i in range(epochs):
            out  = self.forward(X)
            loss = np.mean((out - y) ** 2)
            loss_history.append(loss)
            # Backpropagation (MSE + sigmoid output)
            d_out = (out - y) * out * (1 - out)
            dW3 = self.a2.T @ d_out; db3 = np.sum(d_out, axis=0, keepdims=True)
            d2  = (d_out @ self.W3.T) * self.relu_grad(self.z2)
            dW2 = self.a1.T @ d2;    db2 = np.sum(d2, axis=0, keepdims=True)
            d1  = (d2 @ self.W2.T) * self.relu_grad(self.z1)
            dW1 = X.T @ d1;          db1 = np.sum(d1, axis=0, keepdims=True)
            self.W3 -= lr * dW3; self.b3 -= lr * db3
            self.W2 -= lr * dW2; self.b2 -= lr * db2
            self.W1 -= lr * dW1; self.b1 -= lr * db1
            if i % 100 == 0:
                print(f"        epoch {i:4d}  loss {loss:.5f}")
        return loss_history

print("\nSTEP 9  NeuralNet defined: 30 -> Dense(64,ReLU) -> Dense(32,ReLU) -> Dense(1,Sigmoid)")

# ============================ STEP 10 - Train the Model ======================
print("\nSTEP 10 Training...")
model = NeuralNet(X_train.shape[1])
loss_history = model.train(X_train, y_train.values, lr=0.005, epochs=1000)

plt.figure(figsize=(7, 4))
plt.plot(loss_history, color="steelblue", linewidth=1.5)
plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.title("Training Loss Curve")
plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("figures/05_loss_curve.png", dpi=120); plt.close()

# ============================ STEP 11 - Evaluate Model =======================
probs = model.forward(X_test)
preds = (probs > 0.5).astype(int)
print("\nSTEP 11 Classification Report:")
print(classification_report(y_test, preds, target_names=["Benign", "Malignant"]))
auc = roc_auc_score(y_test, probs)
print(f"        ROC-AUC: {auc:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
cm = confusion_matrix(y_test, preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
axes[0].set_title("Confusion Matrix"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
fpr, tpr, _ = roc_curve(y_test, probs)
axes[1].plot(fpr, tpr, color="tomato", linewidth=2, label=f"AUC = {auc:.3f}")
axes[1].plot([0, 1], [0, 1], "k--", linewidth=1)
axes[1].fill_between(fpr, tpr, alpha=0.1, color="tomato")
axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("figures/06_confusion_roc.png", dpi=120); plt.close()

# ============================ STEP 12 - Output Range Analysis =================
# Core paper methodology (SHERLOCK-style): interval propagation + local search.
def interval_propagation(model, lb, ub):
    """Guaranteed over-approximation of the output range for inputs in [lb, ub]."""
    l, u = lb.copy(), ub.copy()
    W1p, W1n = np.maximum(model.W1, 0), np.minimum(model.W1, 0)
    l1 = np.maximum(W1p.T @ l + W1n.T @ u + model.b1.flatten(), 0)
    u1 = np.maximum(W1p.T @ u + W1n.T @ l + model.b1.flatten(), 0)
    W2p, W2n = np.maximum(model.W2, 0), np.minimum(model.W2, 0)
    l2 = np.maximum(W2p.T @ l1 + W2n.T @ u1 + model.b2.flatten(), 0)
    u2 = np.maximum(W2p.T @ u1 + W2n.T @ l1 + model.b2.flatten(), 0)
    W3p, W3n = np.maximum(model.W3, 0), np.minimum(model.W3, 0)
    l3 = (W3p.T @ l2 + W3n.T @ u2 + model.b3.flatten())[0]
    u3 = (W3p.T @ u2 + W3n.T @ l2 + model.b3.flatten())[0]
    sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    return sig(l3), sig(u3)

def compute_input_gradient(model, x):
    """Numerical gradient of the output w.r.t. input x (Section 4.2 of the paper)."""
    eps = 1e-4; x = x.reshape(1, -1); grad = np.zeros_like(x)
    for j in range(x.shape[1]):
        xp = x.copy(); xp[0, j] += eps
        xm = x.copy(); xm[0, j] -= eps
        grad[0, j] = (model.forward(xp)[0][0] - model.forward(xm)[0][0]) / (2 * eps)
    return grad.flatten()

def local_search(model, x_start, lb, ub, steps=50, lr=0.05, direction="max"):
    """Gradient ascent (max) / descent (min), projected back into P = [lb, ub]."""
    x = x_start.copy(); sign = 1 if direction == "max" else -1
    for _ in range(steps):
        x = np.clip(x + sign * lr * compute_input_gradient(model, x), lb, ub)
    return model.forward(x.reshape(1, -1))[0][0], x

def hybrid_range(model, lb, ub, n_restarts=8, steps=40, delta=0.01):
    """Interval propagation + multi-start local search (Algorithm 1, SHERLOCK)."""
    ip_low, ip_high = interval_propagation(model, lb, ub)
    best_high, best_low = ip_low, ip_high
    for _ in range(n_restarts):
        x0 = np.random.uniform(lb, ub)
        vmax, _ = local_search(model, x0, lb, ub, steps=steps, direction="max")
        if vmax > best_high: best_high = vmax
        vmin, _ = local_search(model, x0, lb, ub, steps=steps, direction="min")
        if vmin < best_low: best_low = vmin
    return {"ip_low": ip_low, "ip_high": ip_high,
            "ls_low": max(best_low - delta, 0.0), "ls_high": min(best_high + delta, 1.0)}

print("\nSTEP 12 Output range analysis on 3 regions (takes a moment)...")
x_ref = X_test[0]; eps1 = 0.10 * np.abs(x_ref) + 0.01
region1 = (x_ref - eps1, x_ref + eps1, "Region 1: Small neighbourhood (eps=0.10)")
hr_idx = int(np.argmax(model.forward(X_test).flatten())); x_hr = X_test[hr_idx]
eps2 = 0.05 * np.abs(x_hr) + 0.01
region2 = (x_hr - eps2, x_hr + eps2, "Region 2: High-risk neighbourhood (eps=0.05)")
region3 = (X_test.min(0), X_test.max(0), "Region 3: Full test set bounding box")

results = []
for lb, ub, label in [region1, region2, region3]:
    r = hybrid_range(model, lb, ub); r["label"] = label; results.append(r)
    print(f"        {label}")
    print(f"          Interval : [{r['ip_low']:.4f}, {r['ip_high']:.4f}]")
    print(f"          Local    : [{r['ls_low']:.4f}, {r['ls_high']:.4f}]")

# ============================ STEP 13 - Visualise Range Analysis =============
fig, ax = plt.subplots(figsize=(9, 4))
ax.axvspan(0.0, 0.5, alpha=0.08, color="steelblue"); ax.axvspan(0.5, 1.0, alpha=0.08, color="tomato")
ax.axvline(0.5, color="black", linewidth=2, linestyle="--", label="Decision boundary")
for i, r in enumerate(results):
    ax.hlines(i + 0.1, r["ip_low"], r["ip_high"], colors="tomato",    linewidth=7, alpha=0.6)
    ax.hlines(i - 0.1, r["ls_low"], r["ls_high"], colors="steelblue", linewidth=7, alpha=0.8)
ax.set_yticks(range(3)); ax.set_yticklabels(["R1", "R2", "R3"]); ax.set_xlim(-0.05, 1.15)
ax.set_xlabel("Network Output (probability)"); ax.set_title("Output Range Intervals"); ax.legend()
plt.tight_layout(); plt.savefig("figures/07_output_ranges.png", dpi=120); plt.close()

# ============================ STEP 14 - Robustness Verification ==============
print("\nSTEP 14 Robustness verification (range must not straddle 0.5):")
BOUNDARY = 0.5
for i, r in enumerate(results):
    lo, hi = r["ls_low"], r["ls_high"]
    crosses = (lo < BOUNDARY) and (hi >= BOUNDARY)
    status  = "SAFE (robust)" if not crosses else "UNSAFE (ambiguous)"
    print(f"        Region {i+1}: [{lo:.4f}, {hi:.4f}] -> {status}")

# ============================ STEP 15 - ReLU Activation Patterns =============
def get_activation_pattern(model, x):
    model.forward(x.reshape(1, -1))
    return np.concatenate([(model.z1[0] > 0).astype(int), (model.z2[0] > 0).astype(int)])

patterns = np.array([get_activation_pattern(model, X_test[i]) for i in range(len(X_test))])
unique_regions = len(np.unique(patterns, axis=0))
print(f"\nSTEP 15 ReLU neurons: {patterns.shape[1]} (64+32) | "
      f"Unique linear regions: {unique_regions}/{len(X_test)}")

avg_act = patterns.mean(axis=0)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("ReLU Activation Patterns - Locally Active Regions", fontweight="bold")
pca_act = PCA(n_components=2).fit_transform(patterns.astype(float))
for label, color, name in [(0, "steelblue", "Benign"), (1, "tomato", "Malignant")]:
    mask = y_test.values == label
    axes[0].scatter(pca_act[mask, 0], pca_act[mask, 1], c=color, label=name,
                    s=25, alpha=0.7, edgecolors="k", linewidths=0.2)
axes[0].set_title("Activation Pattern Space (PCA 2D)"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].bar(range(len(avg_act)), avg_act, color="purple", alpha=0.6)
axes[1].axhline(0.5, color="red", linestyle="--", linewidth=1, label="50% line")
axes[1].set_xlabel("Neuron Index"); axes[1].set_ylabel("Activation Rate")
axes[1].set_title("Neuron Activation Rates"); axes[1].legend(); axes[1].grid(True, alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig("figures/08_relu_patterns.png", dpi=120); plt.close()

# ============================ STEP 16 - Final Summary ========================
acc  = accuracy_score(y_test, preds);  f1   = f1_score(y_test, preds)
prec = precision_score(y_test, preds); rec  = recall_score(y_test, preds)
print("\n" + "=" * 60)
print("  FINAL RESULTS SUMMARY - CT-354 CCP")
print("=" * 60)
print(f"  Samples      : {len(df)} ({len(X_train)} train / {len(X_test)} test)")
print(f"  Architecture : 30 -> Dense(64,ReLU) -> Dense(32,ReLU) -> Dense(1,Sigmoid)")
print(f"  Accuracy {acc:.4f} | Precision {prec:.4f} | Recall {rec:.4f} | F1 {f1:.4f} | AUC {auc:.4f}")
print(f"  Locally active regions: {unique_regions}")
print("=" * 60)

# ============================ EXPORT - learned model for the website =========
# RAW (unscaled) test rows recovered via idx_test, plus the scaler, so the
# website can take a user's raw report values and standardize them the same way.
X_raw = X.values                              # noisy raw features (as used in notebook)
Xte_raw = X_raw[idx_test]
yte = y_test.values.astype(int)
prob_flat = probs.flatten()

layers = [{"W": model.W1.tolist(), "b": model.b1.flatten().tolist()},
          {"W": model.W2.tolist(), "b": model.b2.flatten().tolist()},
          {"W": model.W3.tolist(), "b": model.b3.flatten().tolist()}]

order = np.argsort(prob_flat)
named = {"benign": order[0], "malignant": order[-1], "borderline": order[len(order)//2]}
rng = np.random.default_rng(7)
bi = [i for i in range(len(yte)) if yte[i] == 0]
mi = [i for i in range(len(yte)) if yte[i] == 1]
pick = list(rng.choice(bi, min(25, len(bi)), replace=False)) + \
       list(rng.choice(mi, min(25, len(mi)), replace=False))
rng.shuffle(pick)
pool = [{"values": [round(float(v), 5) for v in Xte_raw[i]], "true": int(yte[i])} for i in pick]

feat = list(X.columns)
ben = X_raw[y.values == 0]; mal = X_raw[y.values == 1]
bundle = {
    "features": feat,
    "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
    "layers": layers,
    "feat_min": X_raw.min(0).tolist(), "feat_max": X_raw.max(0).tolist(),
    "feat_median": np.median(X_raw, 0).tolist(),
    "benign_med": np.median(ben, 0).round(4).tolist(),
    "malign_med": np.median(mal, 0).round(4).tolist(),
    "samples": {k: {"values": [round(float(v), 5) for v in Xte_raw[i]], "true": int(yte[i])}
                for k, i in named.items()},
    "pool": pool,
    "metrics": {"accuracy": round(float(acc), 4), "precision": round(float(prec), 4),
                "recall": round(float(rec), 4), "f1": round(float(f1), 4),
                "auc": round(float(auc), 4), "cm": cm.tolist(),
                "n_train": int(len(X_train)), "n_test": int(len(X_test))},
    "n_total": int(len(df)), "n_mal": int(y.sum()), "n_ben": int((y == 0).sum()),
}
with open("model.js", "w") as f:
    f.write("window.MODEL = " + json.dumps(bundle) + ";")
json.dump(bundle, open("model_bundle.json", "w"))
print(f"\nEXPORT  model.js written ({round(os.path.getsize('model.js')/1024,1)} KB) + figures/ saved.")
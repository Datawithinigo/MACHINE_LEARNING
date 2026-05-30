#!/usr/bin/env python3
# coding: utf-8

# # Assignment 3 - Time Series Analysis
# ## Task 2: Preprocessing & Feature Engineering
# **Dataset:** MIT-BIH Arrhythmia Database (PhysioNet)
# **Goal:** Report discarded beats, analyse per-class feature distributions, perform
# temporal train/test split, apply StandardScaler, build the sliding-window dataset
# for RNN/LSTM, and wrap everything in PyTorch DataLoaders.
#
# This script depends on outputs produced by Task1_DataExploration.py.
# Run Task1_DataExploration.py first to generate mitbih_features_task1.csv.

# ---
# ### 2.0 Imports & Setup

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import wfdb
import joblib
import os
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, LabelEncoder

# Reproducibility
np.random.seed(42)

# I/O directories
data_dir = '/Users/arriazui/Downloads/master/C1_S2/MACHINE_LEARNING/P3/python_code'
img_dir  = os.path.join(data_dir, 'images')
os.makedirs(img_dir, exist_ok=True)

print('Libraries loaded successfully.')


# ---
# ### Re-load shared constants and feature data from Task 1

RECORDS = ['100', '101', '105', '106', '108', '109', '111',
           '112', '115', '117', '119', '201', '213', '219']

FS       = 360
T_WINDOW = 10

FEATURE_COLS = [
    'RR_current', 'RR_prev', 'RR_ratio', 'RR_local_mean',
    'R_amplitude', 'QRS_duration', 'QRS_energy', 'ST_mean'
]

LABEL_MAP = {
    'N': 'N', '.': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
}

CLASS_COLORS = {'N': '#2196F3', 'S': '#FF9800', 'V': '#F44336'}

# Load feature DataFrame saved by Task 1
features_path = os.path.join(data_dir, 'mitbih_features_task1.csv')
df_features = pd.read_csv(features_path)
print(f"Loaded feature matrix: {df_features.shape}")
print(f"Columns: {list(df_features.columns)}")


# ---
# ### 2.1 Discarded Beats Report
#
# Discarded symbols include pacemaker beats (P), non-beat annotations (~, +, |),
# and rhythm-change markers — none of which represent a classifiable individual beat.

discard_rows = []
mapped_symbols  = set(LABEL_MAP.keys())
all_raw_symbols = {}

for rec_id in RECORDS:
    ann = wfdb.rdann(rec_id, 'atr', pn_dir='mitdb')
    n_kept, n_disc = 0, 0
    disc_syms = {}

    for sym in ann.symbol:
        if sym in LABEL_MAP:
            n_kept += 1
        else:
            n_disc += 1
            disc_syms[sym] = disc_syms.get(sym, 0) + 1
            all_raw_symbols[sym] = all_raw_symbols.get(sym, 0) + 1

    discard_rows.append({
        'Record'           : rec_id,
        'Kept'             : n_kept,
        'Discarded'        : n_disc,
        'Discarded symbols': str(disc_syms) if disc_syms else '—'
    })

df_discard  = pd.DataFrame(discard_rows)
total_kept  = df_discard['Kept'].sum()
total_disc  = df_discard['Discarded'].sum()

print("Per-record discard report:")
print(df_discard.to_string(index=False))
print(f"\nTotal kept      : {total_kept:,}")
print(f"Total discarded : {total_disc:,}")
print(f"\nAll discarded symbols across dataset:")
for sym, cnt in sorted(all_raw_symbols.items(), key=lambda x: -x[1]):
    print(f"  '{sym}' : {cnt}")


# ---
# ### 2.2 Per-Class Feature Analysis
#
# We compute the mean value of each feature per class and visualise the full
# distributions as violin plots to assess the discriminative potential of the 8 features.

# Per-class feature means
print("\nPer-class feature means:")
print(df_features.groupby('label')[FEATURE_COLS].mean().round(4).to_string())

# Feature distributions by class (violin plots)
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for idx, feat in enumerate(FEATURE_COLS):
    ax = axes[idx]
    data_plot = [
        df_features.loc[df_features['label'] == cls, feat].values
        for cls in ['N', 'S', 'V']
    ]
    parts = ax.violinplot(data_plot, positions=[0, 1, 2],
                          showmedians=True, showextrema=False)
    for pc, cls in zip(parts['bodies'], ['N', 'S', 'V']):
        pc.set_facecolor(CLASS_COLORS[cls])
        pc.set_alpha(0.7)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.5)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['N', 'S', 'V'], fontsize=10)
    ax.set_title(feat, fontsize=10, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)

fig.suptitle('Feature Distributions by Class\n'
             'Blue = N, Orange = S, Red = V   |   Horizontal line = median',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task2_feature_distributions.png'), dpi=150, bbox_inches='tight')
plt.show()


# ---
# ### 2.3 Train / Test Split (Temporal, Per-Record)
#
# For each record independently, the first 80% of beats (by position) go to the
# training set and the last 20% to the test set.

X_raw = df_features[FEATURE_COLS].values.astype(np.float32)   # (n_beats, 8)
y_str = df_features['label'].values                            # string labels: 'N', 'S', 'V'

# Encode labels as integers: N=0, S=1, V=2 (alphabetical → consistent with LabelEncoder)
le = LabelEncoder()
y  = le.fit_transform(y_str)   # N→0, S→1, V→2
print(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
print(f"\nFull dataset shape : X={X_raw.shape}, y={y.shape}")

# Temporal 80/20 split — applied independently per record
X_train_raw_list, X_test_raw_list = [], []
y_train_list,     y_test_list     = [], []
rec_train_list,   rec_test_list   = [], []
bidx_train_list,  bidx_test_list  = [], []

for rec_id in RECORDS:
    mask  = df_features['record_id'].values == rec_id
    X_rec = X_raw[mask]
    y_rec = y[mask]
    bidx  = df_features['beat_idx'].values[mask]

    # Sort by beat position (safety measure)
    order           = np.argsort(bidx)
    X_rec, y_rec, bidx = X_rec[order], y_rec[order], bidx[order]

    n      = len(X_rec)
    cutoff = int(n * 0.80)

    X_train_raw_list.append(X_rec[:cutoff])
    X_test_raw_list.append(X_rec[cutoff:])
    y_train_list.append(y_rec[:cutoff])
    y_test_list.append(y_rec[cutoff:])
    rec_train_list.append(np.array([rec_id] * cutoff))
    rec_test_list.append(np.array([rec_id] * (n - cutoff)))
    bidx_train_list.append(bidx[:cutoff])
    bidx_test_list.append(bidx[cutoff:])

X_train_raw = np.concatenate(X_train_raw_list)
X_test_raw  = np.concatenate(X_test_raw_list)
y_train     = np.concatenate(y_train_list)
y_test      = np.concatenate(y_test_list)
rec_train   = np.concatenate(rec_train_list)
rec_test    = np.concatenate(rec_test_list)
bidx_train  = np.concatenate(bidx_train_list)
bidx_test   = np.concatenate(bidx_test_list)

print(f"Training set : {X_train_raw.shape[0]:,} beats  (first 80% of each record)")
print(f"Test set     : {X_test_raw.shape[0]:,} beats  (last 20% of each record)")
print()
print('Class distribution — Training set:')
for cls_idx, cls in enumerate(le.classes_):
    cnt = int(np.sum(y_train == cls_idx))
    print(f'  {cls} : {cnt:>6,}  ({100*cnt/len(y_train):.2f}%)')
print()
print('Class distribution — Test set:')
for cls_idx, cls in enumerate(le.classes_):
    cnt = int(np.sum(y_test == cls_idx))
    print(f'  {cls} : {cnt:>6,}  ({100*cnt/len(y_test):.2f}%)')


# ---
# ### 2.4 Feature Scaling with StandardScaler
#
# We fit StandardScaler on the training set only, then transform both splits.
# Fitting on the full dataset would constitute data leakage.

scaler = StandardScaler()
scaler.fit(X_train_raw)   # fitted exclusively on training-record beats

X_train = scaler.transform(X_train_raw).astype(np.float32)
X_test  = scaler.transform(X_test_raw).astype(np.float32)

print('StandardScaler fitted on training records only.')
print(f'  Train mean (should be ~0): {X_train.mean(axis=0).round(3)}')
print(f'  Train std  (should be ~1): {X_train.std(axis=0).round(3)}')
print()
print(f'  Test  mean (may differ)  : {X_test.mean(axis=0).round(3)}')
print(f'  Test  std  (may differ)  : {X_test.std(axis=0).round(3)}')
print()
print('Scaler parameters (per feature):')
for feat, mu, sigma in zip(FEATURE_COLS, scaler.mean_, scaler.scale_):
    print(f'  {feat:<18}  mean={mu:>10.4f}  scale={sigma:>10.4f}')

# Visualise scaling effect for two features
fig, axes = plt.subplots(2, 2, figsize=(7, 7))

for row, feat_idx in enumerate([0, 4]):   # RR_current, R_amplitude
    feat = FEATURE_COLS[feat_idx]

    axes[row, 0].hist(X_train_raw[:, feat_idx], bins=60,
                      color='steelblue', alpha=0.75, edgecolor='white')
    axes[row, 0].set_title(f'{feat}, Before scaling', fontsize=10, fontweight='bold')
    axes[row, 0].set_xlabel('Raw value')
    axes[row, 0].set_ylabel('Count')
    axes[row, 0].grid(True, alpha=0.3)

    axes[row, 1].hist(X_train[:, feat_idx], bins=60,
                      color='darkorange', alpha=0.75, edgecolor='white')
    axes[row, 1].set_title(f'{feat}, After StandardScaler', fontsize=10, fontweight='bold')
    axes[row, 1].set_xlabel('Standardised value (z-score)')
    axes[row, 1].grid(True, alpha=0.3)

fig.suptitle('Effect of StandardScaler', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task2_scaling_effect.png'), dpi=150, bbox_inches='tight')
plt.show()


# ---
# ### 2.5 Sliding Window Dataset for RNN / LSTM
#
# For RNN and LSTM we need sequences of beats rather than individual feature vectors.
# We construct a sliding window of T = 10 consecutive beats:
#   - Each sample: feature vectors for beats i−9 through i → shape (10, 8)
#   - Label: the class of beat i (the final beat in the window)
#   - Stride = 1 (maximum number of training samples)
#
# Windows must not span record boundaries.

def make_windows(X_feat, y_labels, T=T_WINDOW):
    """
    Given a per-beat feature matrix and label array for ONE record,
    return (X_windows, y_windows) where:
        X_windows : shape (n_windows, T, 8)
        y_windows : shape (n_windows,)  label of the LAST beat in each window
    """
    n = len(X_feat)
    if n < T:
        return (np.empty((0, T, X_feat.shape[1]), dtype=np.float32),
                np.empty(0, dtype=np.int64))

    X_win, y_win = [], []
    for i in range(T - 1, n):
        X_win.append(X_feat[i - T + 1 : i + 1])   # beats i-9 to i, shape (T, 8)
        y_win.append(y_labels[i])                   # label of beat i

    return (np.array(X_win, dtype=np.float32),
            np.array(y_win,  dtype=np.int64))


def build_windowed_dataset(X_feat, y_labels, record_ids, beat_idx_arr, T=10):
    X_all_win, y_all_win = [], []
    for rec_id in np.unique(record_ids):
        mask  = record_ids == rec_id
        order = np.argsort(beat_idx_arr[mask])
        X_rec = X_feat[mask][order]
        y_rec = y_labels[mask][order]
        X_w, y_w = make_windows(X_rec, y_rec, T=T)
        if len(X_w) > 0:
            X_all_win.append(X_w)
            y_all_win.append(y_w)
    return (np.concatenate(X_all_win, axis=0),
            np.concatenate(y_all_win, axis=0))


X_train_win, y_train_win = build_windowed_dataset(X_train, y_train, rec_train, bidx_train)
X_test_win,  y_test_win  = build_windowed_dataset(X_test,  y_test,  rec_test,  bidx_test)

# Inspect first window from training set
print("\nFirst window from training set (10 beats × 8 features):")
print(pd.DataFrame(
    X_train_win[0],
    columns=FEATURE_COLS,
    index=[f'beat_{i}' for i in range(T_WINDOW)]
).round(4).to_string())

# Class distribution in windowed dataset
print("\nClass distribution in windowed training set:")
for cls_idx, cls in enumerate(le.classes_):
    cnt = int(np.sum(y_train_win == cls_idx))
    pct = 100 * cnt / len(y_train_win)
    print(f"  {cls} : {cnt:>6,}  ({pct:.2f}%)")

print("\nClass distribution in windowed test set:")
for cls_idx, cls in enumerate(le.classes_):
    cnt = int(np.sum(y_test_win == cls_idx))
    pct = 100 * cnt / len(y_test_win)
    print(f"  {cls} : {cnt:>6,}  ({pct:.2f}%)")

# Visualise one window per class (heatmap of z-scored features)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for cls_idx, cls in enumerate(le.classes_):
    mask        = y_train_win == cls_idx
    example_win = X_train_win[mask][0]
    ax = axes[cls_idx]
    im = ax.imshow(example_win.T, aspect='auto', cmap='RdYlBu_r',
                   interpolation='nearest', vmin=-2, vmax=2)
    ax.set_title(f'Class {cls}', fontsize=10, fontweight='bold',
                 color=CLASS_COLORS[cls])
    ax.set_xlabel('Beat position in window', fontsize=10)
    ax.set_ylabel('Feature', fontsize=10)
    ax.set_yticks(range(len(FEATURE_COLS)))
    ax.set_yticklabels(FEATURE_COLS, fontsize=9)
    ax.set_xticks(range(T_WINDOW))
    ax.set_xticklabels([str(i) for i in range(T_WINDOW - 1)] + [cls], fontsize=10)
    ax.add_patch(plt.Rectangle(
        (T_WINDOW - 1.5, -0.5), 1, len(FEATURE_COLS),
        linewidth=2, edgecolor='black', facecolor='none'
    ))
    plt.colorbar(im, ax=ax, shrink=0.8, label='z-score')

fig.suptitle('Example Sliding Windows by Class (shared z-score scale)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task2_window_examples.png'), dpi=150, bbox_inches='tight')
plt.show()

# Debug: check value ranges per class
for cls_idx, cls in enumerate(le.classes_):
    mask        = y_train_win == cls_idx
    example_win = X_train_win[mask][0]
    print(f"{cls}: min={example_win.min():.2f}, max={example_win.max():.2f}")


# ---
# ### 2.6 PyTorch Dataset & DataLoader
#
# We wrap the windowed arrays in a custom Dataset class and create DataLoader
# objects for use in RNN and LSTM training.

BATCH_SIZE = 128


class ECGWindowDataset(Dataset):
    """
    PyTorch Dataset for the sliding-window ECG beat classification task.

    Parameters
    ----------
    X : np.ndarray, shape (n_windows, T, n_features)
        Scaled feature sequences.
    y : np.ndarray, shape (n_windows,)
        Integer class labels (0=N, 1=S, 2=V).
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = ECGWindowDataset(X_train_win, y_train_win)
test_dataset  = ECGWindowDataset(X_test_win,  y_test_win)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"ECGWindowDataset created:")
print(f"  Training   : {len(train_dataset):,} windows  |  {len(train_loader)} batches of size {BATCH_SIZE}")
print(f"  Test       : {len(test_dataset):,} windows  |  {len(test_loader)} batches")
print()
X_batch, y_batch = next(iter(train_loader))
print(f"Sample batch shapes:")
print(f"  X_batch : {tuple(X_batch.shape)}  (batch_size, T, n_features)")
print(f"  y_batch : {tuple(y_batch.shape)}  (batch_size,)")
print(f"  X dtype : {X_batch.dtype}")
print(f"  y dtype : {y_batch.dtype}")


# ---
# Save arrays, scaler and label encoder for Task 3

np.save(os.path.join(data_dir, 'X_train.npy'),     X_train)
np.save(os.path.join(data_dir, 'X_test.npy'),      X_test)
np.save(os.path.join(data_dir, 'y_train.npy'),     y_train)
np.save(os.path.join(data_dir, 'y_test.npy'),      y_test)

np.save(os.path.join(data_dir, 'X_train_win.npy'), X_train_win)
np.save(os.path.join(data_dir, 'X_test_win.npy'),  X_test_win)
np.save(os.path.join(data_dir, 'y_train_win.npy'), y_train_win)
np.save(os.path.join(data_dir, 'y_test_win.npy'),  y_test_win)

joblib.dump(scaler, os.path.join(data_dir, 'scaler.joblib'))
joblib.dump(le,     os.path.join(data_dir, 'label_encoder.joblib'))

print("\nSaved: X_train.npy, X_test.npy, y_train.npy, y_test.npy")
print("Saved: X_train_win.npy, X_test_win.npy, y_train_win.npy, y_test_win.npy")
print("Saved: scaler.joblib, label_encoder.joblib")

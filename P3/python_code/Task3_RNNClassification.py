#!/usr/bin/env python3
# coding: utf-8

# # Assignment 3 - Time Series Analysis
# ## Task 3: RNN Sequence Classification
# **Dataset:** MIT-BIH Arrhythmia Database (PhysioNet)
# **Goal:** Implement a Vanilla RNN classifier in PyTorch to classify 10-beat sliding
# windows into one of three arrhythmia classes (N, S, V). Train for 20 epochs with
# CrossEntropyLoss and Adam, plot learning curves, and evaluate with macro F1-score
# and a confusion matrix on the held-out test set.
#
# This script depends on outputs produced by Task2_Preprocessing.py.
# Run Task2_Preprocessing.py first to generate the saved .npy and .joblib files.

# ---
# ### 3.0 Imports & Setup

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib
import os

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# I/O directories
data_dir = '/Users/arriazui/Downloads/master/C1_S2/MACHINE_LEARNING/P3/python_code'
img_dir  = os.path.join(data_dir, 'images')
os.makedirs(img_dir, exist_ok=True)

print('Libraries loaded successfully.')


# ---
# ### Load preprocessed data from Task 2

X_train_win = np.load(os.path.join(data_dir, 'X_train_win.npy'))
X_test_win  = np.load(os.path.join(data_dir, 'X_test_win.npy'))
y_train_win = np.load(os.path.join(data_dir, 'y_train_win.npy'))
y_test_win  = np.load(os.path.join(data_dir, 'y_test_win.npy'))

le = joblib.load(os.path.join(data_dir, 'label_encoder.joblib'))

print(f"Training windows : X={X_train_win.shape}, y={y_train_win.shape}")
print(f"Test windows     : X={X_test_win.shape},  y={y_test_win.shape}")
print(f"Classes          : {list(le.classes_)}")


# ---
# ### PyTorch Dataset & DataLoader

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

print(f"Train loader : {len(train_dataset):,} windows  |  {len(train_loader)} batches")
print(f"Test loader  : {len(test_dataset):,} windows  |  {len(test_loader)} batches")


# ---
# ### 3.1 Model Architecture
#
# A Vanilla RNN that processes a 10-beat window (T=10, input_size=8) step by step.
# The hidden state of the final step is fed to a linear layer to produce class logits.

class VanillaRNN(nn.Module):
    """
    Vanilla RNN classifier for beat-level arrhythmia detection.

    Input  : (batch, T=10, input_size=8) — sliding window of beat features
    Output : (batch, num_classes=3)      — raw class scores (logits)
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        # Recurrent layer: processes the 10-beat sequence step by step
        self.rnn    = nn.RNN(input_size, hidden_size, batch_first=True)
        # Output layer: maps final hidden state to 3 class scores
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, _  = self.rnn(x)          # output shape: (batch, 10, hidden_size)
        last_hidden = output[:, -1, :]    # take beat 10's hidden state: (batch, hidden_size)
        return self.linear(last_hidden)   # logits: (batch, 3)


# ---
# ### 3.2 Training Setup & Loop
#
# We train for 20 epochs using CrossEntropyLoss and the Adam optimiser (lr=0.001).
# Both training and validation loss and accuracy are recorded at each epoch for the
# learning curve plots.

model     = VanillaRNN(input_size=8, hidden_size=64, num_classes=3)
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

# Storage for plotting
history = {'train_loss': [], 'val_loss': [],
           'train_acc':  [], 'val_acc':  []}

NUM_EPOCHS = 20

for epoch in range(NUM_EPOCHS):

    # TRAINING PHASE
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for X_batch, y_batch in train_loader:
        optimiser.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimiser.step()

        train_loss    += loss.item()
        preds          = logits.argmax(dim=1)
        train_correct += (preds == y_batch).sum().item()
        train_total   += y_batch.size(0)

    # VALIDATION PHASE
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

            val_loss    += loss.item()
            preds        = logits.argmax(dim=1)
            val_correct += (preds == y_batch).sum().item()
            val_total   += y_batch.size(0)

    # Record & print
    history['train_loss'].append(train_loss / len(train_loader))
    history['val_loss'].append(val_loss   / len(test_loader))
    history['train_acc'].append(train_correct / train_total)
    history['val_acc'].append(val_correct  / val_total)

    print(f"Epoch {epoch+1:>2}/{NUM_EPOCHS} | "
          f"Train loss: {history['train_loss'][-1]:.4f}  acc: {history['train_acc'][-1]:.3f} | "
          f"Val loss: {history['val_loss'][-1]:.4f}  acc: {history['val_acc'][-1]:.3f}")


# ---
# ### 3.3 Learning Curves

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history['train_loss'], label='Train loss')
axes[0].plot(history['val_loss'],   label='Val loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss curves vs epoch')
axes[0].legend()

axes[1].plot(history['train_acc'], label='Train acc')
axes[1].plot(history['val_acc'],   label='Val acc')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy curves vs epoch')
axes[1].legend()

plt.suptitle('Vanilla RNN — Training and Validation Curves', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task3_rnn_learning_curves.png'), dpi=150, bbox_inches='tight')
plt.show()


# ---
# ### 3.4 Evaluation on Test Set
#
# We collect all predictions on the test set and report the per-class classification
# report and confusion matrix.
# Given the strong class imbalance, macro F1 is the primary metric.

model.eval()
all_preds  = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        logits = model(X_batch)
        preds  = logits.argmax(dim=1)   # highest score = predicted class
        all_preds.extend(preds.numpy())
        all_labels.extend(y_batch.numpy())

# Classification report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=['N', 'S', 'V']))

# Confusion matrix
cm   = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['N', 'S', 'V'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap='Blues', ax=ax)
ax.set_title('Vanilla RNN — Confusion Matrix', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task3_rnn_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.show()

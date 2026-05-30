#!/usr/bin/env python3
# coding: utf-8

# # Assignment 3 - Time Series Analysis
# ## Task 1: Data Loading & Exploration
# **Dataset:** MIT-BIH Arrhythmia Database (PhysioNet)
# **Goal:** Load 14 ECG records, apply the AAMI 3-class label mapping, visualise
# ECG segments with annotated beats, analyse class distribution, engineer the 8
# specified features, and produce the feature correlation heatmap.

# ---
# ### 1.0 Library Imports & Setup

# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# PhysioNet data access
import wfdb

# Scikit-learn utilities
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from collections import Counter
import joblib
import os
import torch
from torch.utils.data import Dataset, DataLoader

# Reproducibility
np.random.seed(42)

# Setup output directories
data_dir = '/Users/arriazui/Downloads/master/C1_S2/MACHINE_LEARNING/P3/python_code'
img_dir  = os.path.join(data_dir, 'images')
os.makedirs(img_dir, exist_ok=True)

print('Libraries loaded successfully.')


# ---
# ### 1.1 Dataset Overview — Configuration

# Configuration

RECORDS = ['100', '101', '105', '106', '108', '109', '111',
           '112', '115', '117', '119', '201', '213', '219']

FS        = 360    # sampling rate (Hz) — fixed for MIT-BIH
N_RECORDS = len(RECORDS)
SEED      = 42     # random state for all splits
T_WINDOW  = 10     # sliding window length in beats (for RNN / LSTM)

FEATURE_COLS = [
    'RR_current', 'RR_prev', 'RR_ratio', 'RR_local_mean',
    'R_amplitude', 'QRS_duration', 'QRS_energy', 'ST_mean'
]

# AAMI 3-class label map (as specified in the assignment)
LABEL_MAP = {
    # N: Normal
    'N': 'N', '.': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    # S: Supraventricular ectopic
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    # V: Ventricular ectopic
    'V': 'V', 'E': 'V',
}

# Colour palette consistent across all plots
CLASS_COLORS = {'N': '#2196F3', 'S': '#FF9800', 'V': '#F44336'}

print(f"Records to load  : {N_RECORDS}")
print(f"Records           : {', '.join(RECORDS)}")
print(f"Sampling rate     : {FS} Hz")
print(f"Window size (T)   : {T_WINDOW} beats")
print(f"Random seed       : {SEED}")
print(f"Mapped classes    : N (Normal), S (Supraventricular), V (Ventricular)")


# ---
# ### 1.2 Load All 14 Records
#
# We iterate over all 14 records, loading:
# - The ECG signal (channel 0, MLII lead)
# - The annotation file (R-peak positions + beat symbols)
#
# We apply LABEL_MAP immediately, discarding any unmapped symbols.

records_data = {}   # dict: record_id -> {'signal': ..., 'r_peaks': ..., 'labels': ...}
load_summary = []   # for the summary table

total_discarded = 0
discarded_detail = {}  # record: count of discarded beats

for rec_id in RECORDS:
    # Download record and annotations from PhysioNet
    record = wfdb.rdrecord(rec_id, pn_dir='mitdb')
    ann    = wfdb.rdann(rec_id,    'atr', pn_dir='mitdb')

    signal   = record.p_signal[:, 0]   # channel 0 (MLII lead)
    r_peaks  = ann.sample               # R-peak sample indices
    symbols  = ann.symbol               # raw beat labels

    # Apply label_map, keeping only mapped beats
    mapped_labels, mapped_peaks = [], []
    n_discarded = 0

    for peak, sym in zip(r_peaks, symbols):
        if sym in LABEL_MAP:
            mapped_labels.append(LABEL_MAP[sym])
            mapped_peaks.append(peak)
        else:
            n_discarded += 1

    mapped_labels = np.array(mapped_labels)
    mapped_peaks  = np.array(mapped_peaks)

    total_discarded          += n_discarded
    discarded_detail[rec_id]  = n_discarded

    records_data[rec_id] = {
        'signal'    : signal,
        'r_peaks'   : mapped_peaks,
        'labels'    : mapped_labels,
        'fs'        : record.fs,
        'sig_name'  : record.sig_name,
        'duration_s': len(signal) / record.fs,
    }

    n_counts = {c: int(np.sum(mapped_labels == c)) for c in ['N', 'S', 'V']}
    load_summary.append({
        'Record'      : rec_id,
        'Duration (s)': round(len(signal) / record.fs, 1),
        'Total beats' : len(mapped_labels),
        'N'           : n_counts['N'],
        'S'           : n_counts['S'],
        'V'           : n_counts['V'],
        'Discarded'   : n_discarded,
    })

# Summary table
df_summary = pd.DataFrame(load_summary)
df_summary_totals = df_summary.copy()
totals = df_summary.select_dtypes(include='number').sum()
totals['Record'] = 'TOTAL'
totals['Duration (s)'] = round(totals['Duration (s)'], 1)
df_summary_totals = pd.concat([df_summary, pd.DataFrame([totals])], ignore_index=True)

print(f"\n{'='*65}")
print(f" Records loaded: {N_RECORDS}  |  Total beats discarded: {total_discarded}")
print(f"{'='*65}")
print(df_summary_totals.to_string(index=False))


# ---
# ### Global Beat Counts

all_labels = np.concatenate([records_data[r]['labels'] for r in RECORDS])

n_N     = int(np.sum(all_labels == 'N'))
n_S     = int(np.sum(all_labels == 'S'))
n_V     = int(np.sum(all_labels == 'V'))
n_total = len(all_labels)

print("Global beat counts across all 14 records:")
print(f"  N (Normal)            : {n_N:>6,d}  ({100*n_N/n_total:.1f}%)")
print(f"  S (Supraventricular)  : {n_S:>6,d}  ({100*n_S/n_total:.1f}%)")
print(f"  V (Ventricular)       : {n_V:>6,d}  ({100*n_V/n_total:.1f}%)")
print(f"  ─────────────────────────────────────")
print(f"  TOTAL (kept)          : {n_total:>6,d}")
print(f"  Discarded             : {total_discarded:>6,d}")


# ---
# ### 1.3 ECG Signal Visualisation
#
# We plot 5-second segments of the raw ECG signal for two records — one predominantly
# normal, one with notable arrhythmias. R-peak positions are marked and beat types annotated.

def plot_ecg_segment(record_id, start_sec=5.0, duration_sec=5.0):
    """
    Plot a segment of the ECG signal with R-peak markers and beat-type annotations.

    Parameters
    ----------
    record_id   : str    MIT-BIH record identifier (e.g. '100')
    start_sec   : float  Start of segment in seconds
    duration_sec: float  Duration of segment in seconds
    """
    data    = records_data[record_id]
    signal  = data['signal']
    r_peaks = data['r_peaks']
    labels  = data['labels']
    fs      = data['fs']

    start_sample = int(start_sec * fs)
    end_sample   = min(int((start_sec + duration_sec) * fs), len(signal))

    t_axis = np.arange(start_sample, end_sample) / fs
    seg    = signal[start_sample:end_sample]

    mask       = (r_peaks >= start_sample) & (r_peaks < end_sample)
    seg_peaks  = r_peaks[mask]
    seg_labels = labels[mask]

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(t_axis, seg, color='#1a1a2e', linewidth=0.9, alpha=0.85, label='ECG (MLII)')

    for cls in ['N', 'S', 'V']:
        cls_mask  = seg_labels == cls
        cls_peaks = seg_peaks[cls_mask]
        cls_amps  = signal[cls_peaks]
        cls_times = cls_peaks / fs
        ax.scatter(cls_times, cls_amps,
                   color=CLASS_COLORS[cls], zorder=5, s=60, marker='^',
                   label=f'{cls} beat ({cls_mask.sum()})',
                   edgecolors='white', linewidths=0.5)
        for t_pk, amp_pk, lbl in zip(cls_times, cls_amps, seg_labels[cls_mask]):
            ax.text(t_pk, amp_pk + 0.12, lbl,
                    ha='center', va='bottom', fontsize=8,
                    color=CLASS_COLORS[cls], fontweight='bold')

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Amplitude (mV)', fontsize=11)
    ax.set_title(
        f'Record {record_id}: 5-second ECG segment '
        f'[{start_sec:.0f}s – {start_sec+duration_sec:.0f}s] ',
        fontsize=11, fontweight='bold'
    )
    ax.set_xlim(t_axis[0], t_axis[-1])
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.grid(True, which='minor', alpha=0.1)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, f'ecg_segment_{record_id}.png'), dpi=150, bbox_inches='tight')
    plt.show()


# Plot 1: Record 100
plot_ecg_segment('100', start_sec=10, duration_sec=5)

# Plot 2: Record 119
data_119 = records_data['119']
v_peaks  = data_119['r_peaks'][data_119['labels'] == 'V']
start_v  = max(0, v_peaks[3] / FS - 2)

plot_ecg_segment('119', start_sec=round(start_v, 1), duration_sec=5)

# Plot 3: Record 201 (shows all three classes)
d_201   = records_data['201']
lbl_201 = d_201['labels']
pks_201 = d_201['r_peaks']
v_pks   = pks_201[lbl_201 == 'V']
s_pks   = pks_201[lbl_201 == 'S']

HALF_WIN_S = int(2.5 * FS)

start_201 = None
for vp in v_pks:
    nearby_s = s_pks[(s_pks > vp - HALF_WIN_S) & (s_pks < vp + HALF_WIN_S)]
    if len(nearby_s) > 0:
        centre    = int((vp + nearby_s[0]) // 2)
        start_201 = max(0.0, (centre - HALF_WIN_S) / FS)
        break

if start_201 is None:
    start_201 = max(0.0, v_pks[0] / FS - 2)

plot_ecg_segment('201', start_sec=round(start_201, 1), duration_sec=5)


# ---
# ### 1.4 Class Distribution Analysis
#
# We report the total number of beats per class and produce a bar chart showing the
# distribution across all 14 records.

# Per-record and global class distribution
df_dist = pd.DataFrame(load_summary).set_index('Record')[['N', 'S', 'V', 'Total beats']]

# Add percentage columns
for cls in ['N', 'S', 'V']:
    df_dist[f'{cls}%'] = (df_dist[cls] / df_dist['Total beats'] * 100).round(1)

print("Beat counts per class per record:")
print(df_dist.to_string())
print("\nGlobal totals:")
print(f"  N: {n_N:,} ({100*n_N/n_total:.2f}%)")
print(f"  S: {n_S:,} ({100*n_S/n_total:.2f}%)")
print(f"  V: {n_V:,} ({100*n_V/n_total:.2f}%)")
print(f"  Total: {n_total:,}")


# Class distribution bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: absolute counts
classes = ['N (Normal)', 'S (Supraventricular)', 'V (Ventricular)']
counts  = [n_N, n_S, n_V]
colors  = [CLASS_COLORS['N'], CLASS_COLORS['S'], CLASS_COLORS['V']]

bars = axes[0].bar(classes, counts, color=colors, edgecolor='white',
                   linewidth=1.2, width=0.55)

for bar, count in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 200,
                 f'{count:,}', ha='center', va='bottom',
                 fontsize=10, fontweight='bold')

axes[0].set_title('Beat Count per Class\n(14 records combined)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Number of beats', fontsize=11)
axes[0].set_ylim(0, max(counts) * 1.15)
axes[0].tick_params(axis='x', labelsize=12)
axes[0].grid(True, axis='y', alpha=0.3, linestyle='--')
axes[0].spines[['top', 'right']].set_visible(False)

# Right panel: percentage breakdown per record (stacked bars)
rec_names = list(df_summary['Record'])
n_pct = df_summary['N'] / df_summary['Total beats'] * 100
s_pct = df_summary['S'] / df_summary['Total beats'] * 100
v_pct = df_summary['V'] / df_summary['Total beats'] * 100

x_pos = np.arange(len(rec_names))
w = 0.65

axes[1].bar(x_pos, n_pct, width=w, label='N', color=CLASS_COLORS['N'], edgecolor='white')
axes[1].bar(x_pos, s_pct, width=w, bottom=n_pct, label='S', color=CLASS_COLORS['S'], edgecolor='white')
axes[1].bar(x_pos, v_pct, width=w, bottom=n_pct + s_pct, label='V', color=CLASS_COLORS['V'], edgecolor='white')

axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(rec_names, rotation=45, ha='right', fontsize=12)
axes[1].set_ylabel('Percentage of beats (%)', fontsize=11)
axes[1].set_title('Class Composition per Record\n(stacked %)', fontsize=12, fontweight='bold')
axes[1].set_ylim(0, 110)
axes[1].axhline(100, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
axes[1].legend(loc='lower right', fontsize=12, title='Class')
axes[1].grid(True, axis='y', alpha=0.3, linestyle='--')
axes[1].spines[['top', 'right']].set_visible(False)

plt.suptitle('Class Distribution (14 Records)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task1_class_distribution.png'), dpi=150, bbox_inches='tight')
plt.show()


# ---
# ### 1.5 Feature Engineering
#
# The 8 features specified in the assignment are computed here.
#
# | # | Feature        | Window          | Clinical meaning                           |
# |---|----------------|-----------------|---------------------------------------------|
# | 1 | RR_current     | Beat i          | Current RR interval (ms)                    |
# | 2 | RR_prev        | Beat i−1        | Previous RR interval                         |
# | 3 | RR_ratio       | Beat i          | RR_current / RR_local_mean                   |
# | 4 | RR_local_mean  | Beats i−4 to i  | 5-beat rolling mean RR                       |
# | 5 | R_amplitude    | ±5 samples      | Signal amplitude at the R-peak               |
# | 6 | QRS_duration   | Threshold cross | Width of QRS above 50% of R amplitude        |
# | 7 | QRS_energy     | ±20 samples     | Sum of squared signal values around R-peak   |
# | 8 | ST_mean        | +40 to +120 smp | Mean signal in ST segment                    |

def extract_features_for_record(record_id):
    """
    Extract the 8 specified features for every (mapped) beat in a record.

    Returns a DataFrame with columns:
        record_id, beat_idx, label, and the 8 feature columns.
    """
    data   = records_data[record_id]
    signal = data['signal']
    peaks  = data['r_peaks']
    labels = data['labels']
    fs     = data['fs']
    n      = len(peaks)

    # Compute RR intervals for all consecutive pairs (sample differences → ms)
    rr_intervals = np.diff(peaks) / fs * 1000   # shape (n-1,)

    # Default fallback RR: mean of first 5 available intervals
    fallback_rr = float(np.mean(rr_intervals[:min(5, len(rr_intervals))]))

    rows = []
    for i in range(n):
        peak = peaks[i]

        # RR features
        if i == 0:
            # First beat: no preceding beat → use fallback
            rr_current    = fallback_rr
            rr_prev       = fallback_rr
            rr_local_mean = fallback_rr
        else:
            rr_current = rr_intervals[i - 1]         # interval ending at beat i
            rr_prev    = rr_intervals[i - 2] if i >= 2 else fallback_rr
            # 5-beat rolling mean: use intervals i-4 through i-1 (up to 4 prev)
            start_idx     = max(0, i - 5)
            rr_local_mean = float(np.mean(rr_intervals[start_idx:i]))

        rr_ratio = rr_current / rr_local_mean if rr_local_mean > 0 else 1.0

        # Morphological features
        # R_amplitude: signal value at R-peak
        r_amplitude = float(signal[peak])

        # QRS_duration: width above 50% of R amplitude
        half_amp = 0.5 * r_amplitude
        q_start = peak
        for k in range(peak, max(0, peak - 40), -1):
            if signal[k] < half_amp:
                q_start = k
                break
        q_end = peak
        for k in range(peak, min(len(signal), peak + 40)):
            if signal[k] < half_amp:
                q_end = k
                break
        qrs_duration = float(q_end - q_start)   # in samples

        # QRS_energy: sum of squared samples in ±20 sample window around R-peak
        e_lo = max(0, peak - 20)
        e_hi = min(len(signal), peak + 21)
        qrs_energy = float(np.sum(signal[e_lo:e_hi] ** 2))

        # ST_mean: mean signal from +40 to +120 samples after R-peak
        st_lo = min(len(signal), peak + 40)
        st_hi = min(len(signal), peak + 121)
        st_mean = float(np.mean(signal[st_lo:st_hi])) if st_hi > st_lo else 0.0

        rows.append({
            'record_id'    : record_id,
            'beat_idx'     : i,
            'label'        : labels[i],
            'RR_current'   : rr_current,
            'RR_prev'      : rr_prev,
            'RR_ratio'     : rr_ratio,
            'RR_local_mean': rr_local_mean,
            'R_amplitude'  : r_amplitude,
            'QRS_duration' : qrs_duration,
            'QRS_energy'   : qrs_energy,
            'ST_mean'      : st_mean,
        })

    return pd.DataFrame(rows)


# Extract features for all records
all_dfs = []
for rec_id in RECORDS:
    df_rec = extract_features_for_record(rec_id)
    all_dfs.append(df_rec)
    print(f"  Record {rec_id}: {len(df_rec):,} beats")

df_features = pd.concat(all_dfs, ignore_index=True)

print(f"\nTotal feature matrix shape : {df_features[FEATURE_COLS].shape}")
print(f"Beat label distribution    :")
print(df_features['label'].value_counts().to_string())

# Descriptive statistics of the feature matrix
print("\nDescriptive statistics:")
print(df_features[FEATURE_COLS].describe().round(4).to_string())


# ---
# ### 1.6 Feature Correlation Heatmap
#
# We compute the Pearson correlation matrix of the 8 engineered features and
# visualise it as a heatmap.

# Pearson correlation matrix
corr_matrix = df_features[FEATURE_COLS].corr(method='pearson')
print(corr_matrix.round(3).to_string())

# Heatmap visualisation
fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='RdBu_r',
    vmin=-1, vmax=1,
    center=0,
    square=True,
    linewidths=0.5,
    linecolor='white',
    ax=ax,
    annot_kws={'size': 12},
    cbar_kws={'shrink': 0.8, 'label': 'Pearson r'}
)

ax.set_title('Feature Correlation Heatmap',
             fontsize=12, fontweight='bold', pad=15)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=11)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task1_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.show()

# Identify strongly correlated pairs (|r| > 0.3)
corr_pairs = []
for i in range(len(FEATURE_COLS)):
    for j in range(i+1, len(FEATURE_COLS)):
        r = corr_matrix.iloc[i, j]
        corr_pairs.append({
            'Feature A': FEATURE_COLS[i],
            'Feature B': FEATURE_COLS[j],
            'Pearson r' : round(r, 3),
            '|r|'       : round(abs(r), 3)
        })

df_pairs = pd.DataFrame(corr_pairs).sort_values('|r|', ascending=False)
print("Top correlated feature pairs (|r| > 0.3):")
print(df_pairs[df_pairs['|r|'] > 0.3].to_string(index=False))


# ---
# Save feature DataFrame for use in subsequent tasks

df_features.to_csv(os.path.join(data_dir, 'mitbih_features_task1.csv'), index=False)
print(f"\nSaved: mitbih_features_task1.csv")
print(f"Shape: {df_features.shape}")
print(f"Columns: {list(df_features.columns)}")

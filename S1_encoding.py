"""
Partial correlations
  - partial_corr_S : semantic (GPT)  | all except GPT   
  - partial_corr_A : auditory (Mel)  | all except Mel    
  - partial_corr_V : visual (CORnet) | all except CORnet 
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from pingouin import partial_corr
import matplotlib.pyplot as plt

# ========== Config & Constants ==========
EMBED_DIM = 50
N_TIMES = 161
STEP = 1
N_LAG = 161
N_FOLDS = 10
LAG_TS = np.arange(0, N_TIMES, STEP)   # neural timepoints 0..160
assert len(LAG_TS) == N_LAG

# ========== Subject Info ==========
iSub = 1001  # 1001-1010
iChn = 14    # 0-62
sub_name = f'Sub{iSub}'

if iSub == 1001:
    readSessions = list(range(1, 8))
elif iSub == 1003:
    readSessions = list(range(1, 5))
else:
    readSessions = list(range(1, 7))

# ========== Load Exclusion Vector ==========
exclusion_file = f'./derivatives/MetaData/story_epoch_match/{sub_name}_storyids.npy'
exclude_vector = np.load(exclusion_file)
mask = ~np.isin(exclude_vector, [10, 20, 30])

# ========== Load Neural Data ==========
neural_path = f'./derivatives/eeg_processed/{sub_name}/'
neural_file = f'Chn{iChn}_read.npy'
neural_data = np.load(os.path.join(neural_path, neural_file))  # shape: [samples, timepoints]
print("Neural data shape:", neural_data.shape)
assert neural_data.shape[1] >= N_TIMES, "neural data has fewer timepoints than N_TIMES"

# ========== Load Feature Data ==========
# Semantic: GPT
sem_path = f'./derivatives/Features/semantic/GPT-2'
sem_file = f'{sub_name}_semantic.npy'
sem_emb = np.load(os.path.join(sem_path, sem_file))
print("Semantic (GPT) shape:", sem_emb.shape)

# Visual: CORnet
vis_path = f'./derivatives/Features/Visual'
vis_file = f'{sub_name}_visual.npy'
vis_emb = np.load(os.path.join(vis_path, vis_file))
print("Visual (CORnet) shape:", vis_emb.shape)

# Auditory: Mel
aud_path = f'./derivatives/Features/Auditory/Auditory'
aud_file = f'{sub_name}_auditory.npy'
aud_emb = np.load(os.path.join(aud_path, aud_file))
print("Acoustic (Mel) shape:", aud_emb.shape)

# Word2Vec (50-dim)
w2v_dir = './derivatives/Features/semantic/word2vec'
w2v_file = f'word2vec_pca50_combine_Sub{iSub}_Sess{readSessions[0]}-{readSessions[-1]}_read.npy'
w2v_data = np.load(os.path.join(w2v_dir, w2v_file))
print("Word2Vec shape:", w2v_data.shape)

# ========== Load Syntactic Feature ==========
syn_path = f'./derivatives/Features/Linguistic/Syntactic/binary'
syn_file = f'{sub_name}_syn_combine.npy'
syn_data = np.load(os.path.join(syn_path, syn_file))
print(f"Original Syntactic data shape from file: {syn_data.shape}")

# ========== Load word-level features ==========
wfreq_path = f'./derivatives/Features/Linguistic/Word_frequency/'
wfreq_file = f'{sub_name}_logfreq_combine.npy'
wfreq_data = np.load(os.path.join(wfreq_path, wfreq_file))
print(f"Original Word frequency shape from file: {wfreq_data.shape}")

if syn_data.ndim == 1:
    syn_data = syn_data.reshape(-1, 1)
if wfreq_data.ndim == 1:
    wfreq_data = wfreq_data.reshape(-1, 1)

print(f"--> Using {syn_data.shape[1]} dimensions for Syntactic feature.")
print(f"--> Using {wfreq_data.shape[1]} dimensions for Frequency feature.")

# ========== Visual events: char-length onset / offset impulse bases ==========
# shape (n_words, 161, 5); offset file exists only if the presentation had gaps
vis_dir = ('./derivatives/Features/Visual_events')
X_on = np.load(os.path.join(vis_dir, f'{sub_name}_charlen_onset_basis.npy'))
off_path = os.path.join(vis_dir, f'{sub_name}_charlen_offset_basis.npy')
has_off = os.path.exists(off_path)
X_off = np.load(off_path) if has_off else None
print(f"X_on shape: {X_on.shape}   X_off: "
      f"{X_off.shape if has_off else 'none (no gaps)'}")
assert X_on.shape[1] == N_LAG, "timing time-axis must equal N_LAG (161)"

# ---- flatten into one static feature: (n_words, 161, 5[+5]) -> (n_words, 1610) ----
# column layout: for each of the 161 time samples, the 5 onset bins
# followed by the 5 offset bins (lag-major, bin-minor).
if has_off:
    assert X_off.shape == X_on.shape, "X_on and X_off must have the same shape"
    X_evt = np.concatenate([X_on, X_off], axis=2)      # (n_words, 161, 10)
else:
    X_evt = X_on                                        # (n_words, 161, 5)
X_evt = X_evt.reshape(X_evt.shape[0], -1)               # (n_words, 1610) or (n_words, 805)
print("Flattened visual-event shape:", X_evt.shape)

# ========== Align rows ==========
n_words = sem_emb.shape[0]
for name, arr in [('CORnet', vis_emb), ('mel', aud_emb), ('w2v', w2v_data),
                  ('syn', syn_data), ('freq', wfreq_data), ('vis_event', X_evt),
                  ('neural', neural_data), ('mask', mask)]:
    assert arr.shape[0] == n_words, f"{name} has {arr.shape[0]} rows, expected {n_words}"

# ========== Concatenate, Mask, then Standardize ==========
# Feature layout: GPT | CORnet | EVT | Mel | W2V | Syn | Freq
blocks = {
    'GPT':  sem_emb,
    'CORnet': vis_emb,
    'EVT':  X_evt,
    'MEL':  aud_emb,
    'W2V':  w2v_data,
    'SYN':  syn_data,
    'FREQ': wfreq_data,
}
X_full = np.hstack(list(blocks.values()))
X = X_full[mask]

scaler = StandardScaler()
X = scaler.fit_transform(X)

print(f"Applied mask: excluded {np.sum(~mask)} samples, kept {np.sum(mask)} samples")
print(f"Final combined feature matrix shape (X): {X.shape}")

# ========== Feature column indices ==========
cols, start = {}, 0
for name, arr in blocks.items():
    cols[name] = np.arange(start, start + arr.shape[1])
    start += arr.shape[1]
assert start == X.shape[1]

def pick(*names):
    return np.concatenate([cols[n] for n in names])

models = {
    # ---- target models (unique part of interest) ----
    'S':  pick('GPT'),
    'A':  pick('MEL'),
    'V':  pick('CORnet'),                                       # visual = CORnet only

    # ---- control models ("everything else", EVT always included) ----
    'NS': pick('CORnet', 'EVT', 'MEL', 'W2V', 'SYN', 'FREQ'),  # All_Except_S
    'NA': pick('GPT', 'CORnet', 'EVT', 'W2V', 'SYN', 'FREQ'),  # All_Except_A
    'NV': pick('GPT', 'EVT', 'MEL', 'W2V', 'SYN', 'FREQ'),   # All_Except_V (drops CORnet, keeps EVT)
}
for name, idx in models.items():
    print(f"  model {name:3s}: {len(idx)} features")

# target model -> control model
partial_specs = {
    'S': 'NS',
    'A': 'NA',
    'V': 'NV',
}

# ========== Cross-validation Setup ==========
kf = KFold(n_splits=N_FOLDS)

# ========== Initialize Results ==========
corr_results = {f'partial_corr_{tgt}': np.zeros((N_LAG, N_FOLDS))
                for tgt in partial_specs}

# ========== Main Loop: time point by time point ==========
for iLag, t in enumerate(LAG_TS):
    print(f"Time lag {iLag}, neural timepoint {t}")
    y_all = neural_data[:, t]
    y_target = y_all[mask]

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_target[train_idx], y_target[test_idx]

        # Target and control models
        preds = {}
        for name, idx in models.items():
            model = LinearRegression().fit(X_train[:, idx], y_train)
            preds[name] = model.predict(X_test[:, idx])

        # DataFrame for partial correlations
        df = pd.DataFrame({'y_test': y_test,
                           **{f'y_pred_{name}': p for name, p in preds.items()}})

        # Partial correlation for each modality controlling for everything else
        for tgt, ctrl in partial_specs.items():
            corr_results[f'partial_corr_{tgt}'][iLag, fold] = partial_corr(
                df, x=f'y_pred_{tgt}', y='y_test', covar=f'y_pred_{ctrl}'
            )['r'].values[0]

# ========== Save ==========
save_dir = f'./Results/{sub_name}'
os.makedirs(save_dir, exist_ok=True)
out_path = os.path.join(save_dir, f'{sub_name}_Combine_Ch{iChn}_partial_SVA.npz')
np.savez(out_path, **corr_results)
print("Results saved to:", out_path)

# ========== Plot ==========
Plot_figure = 0
if Plot_figure == 1:
    time_points = np.arange(-2, 2 + 0.025, 0.025)

    plt.figure(figsize=(10, 6))
    plt.plot(time_points, np.nanmean(corr_results['partial_corr_S'], 1),
             label='Semantic | rest', color='gold')
    plt.plot(time_points, np.nanmean(corr_results['partial_corr_V'], 1),
             label='Visual (CORnet) | rest', color='red')
    plt.plot(time_points, np.nanmean(corr_results['partial_corr_A'], 1),
             label='Auditory | rest', color='blue')

    plt.xlabel('Time (s)')
    plt.ylabel('Partial correlation r')
    plt.title(f'Sub{iSub} Ch{iChn} Partial Correlations (10-fold CV Mean)')
    plt.legend()
    plt.tight_layout()

    plot_dir = './1-test_figure'
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f'Sub{iSub}_Ch{iChn}_partial_SVA_Read.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plot saved to: {plot_path}")
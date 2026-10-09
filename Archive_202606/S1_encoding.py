import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from pingouin import partial_corr
import matplotlib.pyplot as plt

# ========== Config & Constants ==========
EMBED_DIM = 50
N_TIMES = 161
STEP = 1
N_LAG = 161
N_FOLDS = 10

# ========== Subject Info ==========
iSub = 1001 # 1001-1010
iChn = 14 # 0-62
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

# ========== Load Feature Data ==========
# Semantic: GPT
sem_path = f'./derivatives/Features/semantic/GPT-2'
sem_file = f'{sub_name}_semantic.npy' 
sem_emb = np.load(os.path.join(sem_path, sem_file))
print("Semantic (GPT) shape:", sem_emb.shape) 

# Visual: VWFA
vis_path = f'./derivatives/Features/Visual' 
vis_file = f'{sub_name}_visual.npy' 
vis_emb = np.load(os.path.join(vis_path, vis_file)) 
print("Visual (VWFA) shape:", vis_emb.shape) 

# Auditory: Mel 
aud_path = f'./derivatives/Features/Auditory/Auditory' # Renamed path variable
aud_file = f'{sub_name}_auditory.npy'
aud_emb = np.load(os.path.join(aud_path, aud_file)) 
print("Acoustic (Mel) shape:", aud_emb.shape) 

# Word2Vec (50-dim)
w2v_dir = './derivatives/Features/semantic/word2vec'
w2v_file = f'word2vec_pca50_combine_Sub{iSub}_Sess{readSessions[0]}-{readSessions[-1]}_read.npy'
w2v_data = np.load(os.path.join(w2v_dir, w2v_file))
print("Word2Vec shape:", w2v_data.shape)

# ========== Load Timing Features ==========
timing_path = f'./derivatives/Features/Timing/{sub_name}_timing_features.npz'
timing_raw = np.load(timing_path, allow_pickle=True)

word_onsets    = timing_raw['onsets']          # current word onset
word_offsets   = timing_raw['offsets']         # current word offset
word_durations = timing_raw['durations']       # current word duration

# Previous word timing (shift by 1, pad first entry with 0)
prev_onsets    = np.concatenate([[0], word_onsets[:-1]])
prev_offsets   = np.concatenate([[0], word_offsets[:-1]])

# Stack into timing feature matrix [samples, 5]
timing_data = np.column_stack([
    word_onsets, word_offsets, word_durations,
    prev_onsets, prev_offsets
])
print("Timing shape:", timing_data.shape)

# ========== Load Syntactic Feature ==========
syn_path = f'./derivatives/Features/Linguistic/Syntactic/'
syn_file = f'{sub_name}_syn_combine.npy'
syn_data = np.load(os.path.join(syn_path, syn_file))
print(f"Original Syntactic data shape from file: {syn_data.shape}")

# ========== Load word-level features ==========
wfreq_path = f'./derivatives/Features/Linguistic/Word_frequency/'
wfreq_file = f'{sub_name}_logfreq_combine.npy'
wfreq_data = np.load(os.path.join(wfreq_path, wfreq_file))
print(f"Original Word frequency shape from file: {wfreq_data.shape}")

# ========== Concatenate, Mask, then Standardize ==========
if syn_data.ndim == 1: 
    syn_data = syn_data.reshape(-1, 1)
if wfreq_data.ndim == 1: 
    wfreq_data = wfreq_data.reshape(-1, 1)

print(f"--> Using {syn_data.shape[1]} dimensions for Syntactic feature.")
print(f"--> Using {wfreq_data.shape[1]} dimensions for Frequency feature.")

# Feature layout: GPT | VWFA | Mel | W2V | Timing | Syn | Freq
X_full = np.hstack([sem_emb, vis_emb, aud_emb, w2v_data, timing_data, syn_data, wfreq_data])
X = X_full[mask]

scaler = StandardScaler()
X = scaler.fit_transform(X)

print(f"Applied mask: excluded {np.sum(~mask)} samples, kept {np.sum(mask)} samples")
print(f"Final combined feature matrix shape (X): {X.shape}")

# ========== Feature Slices ==========
idx_S = 0
idx_V = idx_S + sem_emb.shape[1]
idx_A = idx_V + vis_emb.shape[1]
idx_W = idx_A + aud_emb.shape[1]
idx_T = idx_W + timing_data.shape[1]
idx_Syn = idx_T + syn_data.shape[1]
idx_Freq = idx_Syn + wfreq_data.shape[1]
idx_End = X_full.shape[1]

# Define slices for Semantic (S), Visual (V), and Auditory (A) vs. Everything Else
slices = {
    'S': np.r_[idx_S:idx_V],
    'All_Except_S': np.r_[idx_V:idx_End],
    
    'V': np.r_[idx_V:idx_A],
    'All_Except_V': np.r_[idx_S:idx_V, idx_A:idx_End],
    
    'A': np.r_[idx_A:idx_W], 
    'All_Except_A': np.r_[idx_S:idx_A, idx_W:idx_End]
}

# ========== Cross-validation Setup ==========
kf = KFold(n_splits=N_FOLDS)

# ========== Initialize Results ==========
corr_results = {key: np.zeros((N_LAG, N_FOLDS)) for key in [
    'corr_full',
    'corr_S', 'corr_All_Except_S', 'partial_corr_S',
    'corr_V', 'corr_All_Except_V', 'partial_corr_V',
    'corr_A', 'corr_All_Except_A', 'partial_corr_A'
]}

# ========== Main Loop ==========
for iLag, t in enumerate(range(0, N_TIMES + STEP, STEP)):
    print(f"Time lag {iLag}, neural timepoint {t}")
    y_all = neural_data[:, t]
    y_target = y_all[mask]

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_target[train_idx], y_target[test_idx]

        # Full model
        model_full = LinearRegression().fit(X_train, y_train)
        y_pred_full = model_full.predict(X_test)
        corr_results['corr_full'][iLag, fold] = pearsonr(y_test, y_pred_full)[0]

        # Individual and "All Except" models
        preds = {}
        for key, sl in slices.items():
            model = LinearRegression().fit(X_train[:, sl], y_train)
            preds[key] = model.predict(X_test[:, sl])
            corr_results[f'corr_{key}'][iLag, fold] = pearsonr(y_test, preds[key])[0]

        # DataFrame for partial correlations
        df = pd.DataFrame({
            'y_test':              y_test,
            'y_pred_S':            preds['S'],
            'y_pred_All_Except_S': preds['All_Except_S'],
            'y_pred_V':            preds['V'],
            'y_pred_All_Except_V': preds['All_Except_V'],
            'y_pred_A':            preds['A'],
            'y_pred_All_Except_A': preds['All_Except_A']
        })

        # Partial correlations for each modality controlling for everything else
        corr_results['partial_corr_S'][iLag, fold] = partial_corr(
            df, x='y_pred_S', y='y_test', covar='y_pred_All_Except_S'
        )['r'].values[0]

        corr_results['partial_corr_V'][iLag, fold] = partial_corr(
            df, x='y_pred_V', y='y_test', covar='y_pred_All_Except_V'
        )['r'].values[0]

        corr_results['partial_corr_A'][iLag, fold] = partial_corr(
            df, x='y_pred_A', y='y_test', covar='y_pred_All_Except_A'
        )['r'].values[0]

# ========== Save ==========
save_dir = f'./Results/{sub_name}'
os.makedirs(save_dir, exist_ok=True)
out_path = os.path.join(save_dir, f'{sub_name}_Combine_Ch{iChn}_corr.npz')
np.savez(out_path, **corr_results)
print("Results saved to:", out_path)

# ========== Plot ==========
Plot_figure = 0
if Plot_figure == 1:
    time_points = np.arange(-2, 2 + 0.025, 0.025)

    plt.figure(figsize=(10, 6))
    
    # Plotting the means across the 10 folds
    plt.plot(time_points, corr_results['partial_corr_S'].mean(axis=1), label='Partial Corr Semantic', color='green')
    plt.plot(time_points, corr_results['partial_corr_V'].mean(axis=1), label='Partial Corr Visual', color='red')
    plt.plot(time_points, corr_results['partial_corr_A'].mean(axis=1), label='Partial Corr Auditory', color='blue')
    plt.plot(time_points, corr_results['corr_full'].mean(axis=1), label='Full Model Corr', color='black', alpha=0.5, linestyle=':')

    plt.xlabel('Time (s)')
    plt.ylabel('Correlation r')
    plt.title(f'Sub{iSub} Ch{iChn} Correlations (10-fold CV Mean)')
    plt.legend()
    plt.tight_layout()

    plot_dir = './1-test_figure'
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f'Sub{iSub}_Ch{iChn}_correlations_Read.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plot saved to: {plot_path}")
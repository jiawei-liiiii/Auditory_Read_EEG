import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from pingouin import partial_corr
import warnings
warnings.filterwarnings('ignore')

# ===== Config & Constants =====
N_TIMES = 820
STEP = 5
N_LAG = 161
N_FOLDS = 10
RANDOM_SEED = 42

# ========== Subject Info ==========
iSub = 1001 # 1001-1010
iChn = 14 # 0-62
sub_name = f'Sub{iSub}'

hop_configs = [10]

if iSub == 1001:
    readSessions = list(range(1, 8))
elif iSub == 1003:
    readSessions = list(range(1, 5))
else:
    readSessions = list(range(1, 7))

print(f"Processing {sub_name}, Channel {iChn}")

# ========== Load Neural Data ==========
neural_path = f'./derivatives/eeg_processed/{sub_name}/'
neural_file = f'Chn{iChn}_read.npy'
neural_data = np.load(os.path.join(neural_path, neural_file))  # shape: [samples, timepoints]
print("Neural data shape:", neural_data.shape)


# ========== Load Control Features ==========
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


# ========== Load Exclusion Vector ==========
exclusion_file = f'./derivatives/MetaData/story_epoch_match/{sub_name}_storyids.npy'
exclude_vector = np.load(exclusion_file)
mask = ~np.isin(exclude_vector, [10, 20, 30])

# Apply exclusion to static data
neural_data_masked = neural_data[mask]
sem_emb_masked = sem_emb[mask]
vis_emb_masked = vis_emb[mask]
w2v_data_masked = w2v_data[mask]      
timing_data_masked = timing_data[mask]
syn_data_masked = syn_data[mask]
wfreq_data_masked = wfreq_data[mask]

# ===== Scale static features =====
print("Scaling static features...")
scaler_semantic = StandardScaler()
sem_emb_scaled = scaler_semantic.fit_transform(sem_emb_masked)

scaler_visual = StandardScaler()
vis_emb_scaled = scaler_visual.fit_transform(vis_emb_masked)

scaler_w2v = StandardScaler()                                  
w2v_data_scaled = scaler_w2v.fit_transform(w2v_data_masked)    

scaler_timing = StandardScaler()
timing_data_scaled = scaler_timing.fit_transform(timing_data_masked)

scaler_syn = StandardScaler()
syn_data_scaled = scaler_syn.fit_transform(syn_data_masked)

scaler_wfreq = StandardScaler()
wfreq_data_scaled = scaler_wfreq.fit_transform(wfreq_data_masked)

# ===== Combine All Control Features =====
print("Concatenating all control features into a single block...")
X_control_all = np.concatenate([
    sem_emb_scaled,
    vis_emb_scaled,
    w2v_data_scaled,       
    timing_data_scaled,
    syn_data_scaled,
    wfreq_data_scaled
], axis=1)

print(f"✅ Combined control features shape: {X_control_all.shape}")

# ==========================================================
# ===== Main Hop Length Loop ===============================
# ==========================================================
for hop_ms in hop_configs:
    print(f"\n================ Processing Hop Length: {hop_ms}ms ================")
    
    base_dir = Path('./derivatives/Features/Auditory/hop_length')
    mel_file_path = base_dir / sub_name / 'bin_pca' / f'{sub_name}_hop{hop_ms}ms_bin_pca.npy'

    if not mel_file_path.exists():
        print(f"⚠️ Warning: File not found at {mel_file_path}. Skipping.")
        continue

    mel_data = np.load(mel_file_path)
    print(f"✅ Mel shape for {hop_ms}ms:", mel_data.shape)

    if mel_data.shape[0] != neural_data.shape[0]:
        print(f"⚠️ Mismatch between samples! Neural: {neural_data.shape[0]}, Mel: {mel_data.shape[0]}. Skipping.")
        continue
    
    # Mask the hop-specific mel data
    mel_data_masked = mel_data[mask]
    N_TimeBins = mel_data_masked.shape[2]
    
    print(f"✅ After masking - samples: {mel_data_masked.shape[0]}")
    print(f"✅ Number of time bins for {hop_ms}ms: {N_TimeBins}")

    # Initialize Result Dict for this specific hop_ms
    corr_results = np.zeros((N_LAG, N_FOLDS, N_TimeBins))

    # ===== Main Regression Loop =====
    print(f"Starting partial correlation analysis for {hop_ms}ms...")

    for iTime in range(N_TimeBins):
        if (iTime + 1) % 10 == 0 or iTime == 0:
            print(f"Processing time bin {iTime + 1}/{N_TimeBins}")
        
        # Scale acoustic features for current time bin
        acoustic_timebin = mel_data_masked[:, :, iTime]
        scaler_acoustic = StandardScaler()
        acoustic_timebin_scaled = scaler_acoustic.fit_transform(acoustic_timebin)
        
        for iLag, t in enumerate(range(20, N_TIMES, STEP)):
            if t >= neural_data_masked.shape[1]:
                break
                
            # Get features and neural data
            X_control = X_control_all
            X_acoustic = acoustic_timebin_scaled
            y = neural_data_masked[:, t]
            
            if len(y) < N_FOLDS:
                continue
            
            # Cross-validation
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
            for fold, (train_idx, test_idx) in enumerate(kf.split(X_control)):
                
                # Split data
                X_C_train, X_C_test = X_control[train_idx], X_control[test_idx]
                X_A_train, X_A_test = X_acoustic[train_idx], X_acoustic[test_idx]
                
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Train the unified control model and the acoustic model
                model_C = LinearRegression().fit(X_C_train, y_train)
                model_A = LinearRegression().fit(X_A_train, y_train)
                
                # Get predictions
                y_pred_C = model_C.predict(X_C_test)
                y_pred_A = model_A.predict(X_A_test)
                
                # Create DataFrame for partial correlation analysis
                df = pd.DataFrame({
                    'y_test': y_test,
                    'y_pred_A': y_pred_A,
                    'y_pred_control': y_pred_C
                })
                
                # Calculate partial correlation: Acoustic controlling for the combined control prediction
                try:
                    pc_result = partial_corr(
                        df, 
                        x='y_test', 
                        y='y_pred_A', 
                        covar=['y_pred_control']
                    )
                    corr_val = pc_result['r'].values[0]
                    if np.isnan(corr_val):
                        corr_val = 0.0
                    corr_results[iLag, fold, iTime] = corr_val
                except Exception as e:
                    corr_results[iLag, fold, iTime] = 0.0

    print(f"✅ Partial correlation analysis for {hop_ms}ms completed!")

    # ===== Save Results =====
    save_dir = Path(f'./Results/{sub_name}')
    save_dir.mkdir(parents=True, exist_ok=True)

    save_file = save_dir / f'{sub_name}_PartialCorr_Ch{iChn}_hop{hop_ms}ms.npy'
    np.save(save_file, corr_results)
    print(f"✅ Results saved to: {save_file}")

    # ========== Plot Line Plot (2x5 Subplots) ==========
    plot_figure = 0
    if plot_figure == 1:
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8), sharey=True)
        axes_flat = axes.flatten()

        # Average across folds for plotting -> shape becomes (N_LAG, N_TimeBins)
        mean_corr = np.mean(corr_results, axis=1)
        lag_times = np.arange(N_LAG)

        # Select exactly 10 time bins to perfectly fill the 2x5 grid safely
        if N_TimeBins >= 10:
            time_bins_to_plot = np.linspace(0, N_TimeBins - 1, 10, dtype=int)
        else:
            time_bins_to_plot = range(N_TimeBins)

        for i, iTime in enumerate(time_bins_to_plot):
            if i >= 10:
                break
                
            ax = axes_flat[i]
            # Plot the specific time bin column from mean_corr
            ax.plot(lag_times, mean_corr[:, iTime], label=f'TimeBin {iTime+1}', color='tab:blue')
            ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
            ax.set_title(f'TimeBin {iTime+1}')
            
            if i >= 5: # Bottom row
                ax.set_xlabel('Lag Index')
            if i % 5 == 0: # First column
                ax.set_ylabel('Partial Corr (r)')

        fig.suptitle(f'Mel Encoding Lags by TimeBin (Channel {iChn} | Hop: {hop_ms}ms) | {sub_name}', fontsize=16)
        plt.tight_layout()
        
        fig_save_path = Path('./test_figures_partial')
        fig_save_path.mkdir(parents=True, exist_ok=True)
        fig_out_path = fig_save_path / f'{sub_name}_Combine_Ch{iChn}_hop{hop_ms}ms_bin_pca.png'
        plt.savefig(fig_out_path, dpi=300)
        print(f"✅ {hop_ms}ms Plot saved to: {fig_out_path}")
        
        plt.close(fig)

    # ===== Summary Statistics =====
    print(f"\n=== Summary Statistics for {hop_ms}ms ===")
    print(f"Total samples used: {mel_data_masked.shape[0]}")
    print(f"Total time bins processed: {N_TimeBins}")
    print(f"Result shape: {corr_results.shape}")

    mean_partial_corr = np.mean(corr_results, axis=(1, 2))
    print(f"Mean partial correlation across all data: {np.mean(mean_partial_corr):.4f}")

    best_lag = np.argmax(mean_partial_corr)
    print(f"Best lag: {best_lag} (r={mean_partial_corr[best_lag]:.4f})")

    overall_mean = np.mean(corr_results)
    overall_std = np.std(corr_results)
    print(f"Overall mean +/- std: {overall_mean:.4f} +/- {overall_std:.4f}")

print("\n✅ All hop lengths completed successfully!")
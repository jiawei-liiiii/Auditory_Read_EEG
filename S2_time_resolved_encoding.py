import os
import sys
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
N_LAG = 161
N_FOLDS = 10
RANDOM_SEED = 42

# ===== Subject Info =====
iSub = 1001 # 1001-1010
iChn = 14 # 0-62
sub_name = f'Sub{iSub}'

# ===== Session Info (minimal logic) =====
if iSub == 1001:
    readSessions = list(range(1, 8))
elif iSub == 1002:
    readSessions = list(range(1, 7))
elif iSub == 1003:
    readSessions = list(range(1, 5))
else:
    readSessions = list(range(1, 7))

print(f"Processing {sub_name}, Channel {iChn}")

# ===== Load Neural Data =====
neural_path = f'./derivatives/eeg_processed/{sub_name}/'
neural_file = f'{sub_name}_Ch{iChn}.npy'
neural_data = np.load(os.path.join(neural_path, neural_file))  # shape: [samples, timepoints]
print("Neural data shape:", neural_data.shape)

# ===== Load Auditory Features  =====
aud_path = f'./derivatives/Features/auditory_time_resolved/'
aud_file = f'{sub_name}_aud_time_res.npy'
aud_emb = np.load(os.path.join(aud_path, aud_file))  
print(f"✅ Auditory features shape: {aud_emb.shape}")

# Get number of time bins from mel data
N_TimeBins = aud_emb.shape[2]
print(f"✅ Number of time bins: {N_TimeBins}")

# ========== Load Control Features ==========
# Semantic: GPT
sem_path = f'./derivatives/Features/semantic'
sem_file = f'{sub_name}_semantic.npy' 
sem_emb = np.load(os.path.join(sem_path, sem_file))
print("Semantic (GPT) shape:", sem_emb.shape) 

# Visual: VWFA
vis_path = f'./derivatives/Features/visual' 
vis_file = f'{sub_name}_visual.npy' 
vis_emb = np.load(os.path.join(vis_path, vis_file)) 
print("Visual (VWFA) shape:", vis_emb.shape) 

# ===== Load Exclusion Vector and Apply Mask =====
exclusion_file = f'./derivatives/MetaData/storyInd/{sub_name}_storyids.npy'
exclude_vector = np.load(exclusion_file)
mask = ~np.isin(exclude_vector, [10, 20, 30])

# Apply exclusion to all data
aud_emb_masked = aud_emb[mask]
neural_data_masked = neural_data[mask]
sem_emb_masked = sem_emb[mask]
vis_emb_masked = vis_emb[mask]

# ===== Scale features =====
print("Scaling features...")

# Scale semantic features
scaler_semantic = StandardScaler()
sem_emb_scaled = scaler_semantic.fit_transform(sem_emb_masked)

# Scale visual features
scaler_visual = StandardScaler()
vis_emb_scaled = scaler_visual.fit_transform(vis_emb_masked)

print(f"✅ Auditory features scaled: {aud_emb_masked.shape}")
print(f"✅ Neural data scaled: {neural_data_masked.shape}")
print(f"✅ Semantic features scaled: {sem_emb_scaled.shape}")
print(f"✅ Visual features scaled: {vis_emb_scaled.shape}")

# ===== Initialize Result Dict =====
corr_results =  np.zeros((N_LAG, N_FOLDS, N_TimeBins))

# ===== Main Regression Loop =====
print("Starting partial correlation analysis...")

for iTime in range(N_TimeBins):
    print(f"Processing time bin {iTime + 1}/{N_TimeBins}")
    
    # Scale auditory features for current time bin
    aud_timebin = aud_emb_masked[:, :, iTime]
    scaler_aud = StandardScaler()
    aud_timebin_scaled = scaler_aud.fit_transform(aud_timebin)
    
    for iLag in range(0, N_LAG):
            
        # Get features and neural data
        X_semantic = sem_emb_scaled
        X_visual = vis_emb_scaled
        X_auditory = aud_timebin_scaled
        y = neural_data_masked[:, iLag]
        
        if len(y) < N_FOLDS:
            continue
        
        # Cross-validation
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_semantic)):
            
            # Split data
            X_S_train, X_S_test = X_semantic[train_idx], X_semantic[test_idx]
            X_V_train, X_V_test = X_visual[train_idx], X_visual[test_idx]
            X_A_train, X_A_test = X_auditory[train_idx], X_auditory[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train separate models for each modality
            model_S = LinearRegression().fit(X_S_train, y_train)
            model_V = LinearRegression().fit(X_V_train, y_train)  
            model_A = LinearRegression().fit(X_A_train, y_train)
            
            # Get predictions
            y_pred_S = model_S.predict(X_S_test)
            y_pred_V = model_V.predict(X_V_test)
            y_pred_A = model_A.predict(X_A_test)
            
            # Create DataFrame for partial correlation analysis
            df = pd.DataFrame({
                'y_test': y_test,
                'y_pred_S': y_pred_S,
                'y_pred_V': y_pred_V,
                'y_pred_A': y_pred_A
            })
            
            # Calculate partial correlation: auditory controlling for Semantic+Visual
            try:
                pc_result = partial_corr(df, x='y_test', y='y_pred_A', covar=['y_pred_S', 'y_pred_V'])
                corr_val = pc_result['r'].values[0]
                if np.isnan(corr_val):
                    corr_val = 0.0
                corr_results[iLag, fold, iTime] = corr_val
            except Exception as e:
                # Handle cases where partial correlation fails
                corr_results[iLag, fold, iTime] = 0.0

print("✅ Partial correlation analysis completed!")

# ===== Save Results =====
save_dir = f'./test_data/2-time_resolved/{sub_name}'
os.makedirs(save_dir, exist_ok=True)

save_file = os.path.join(save_dir, f'{sub_name}_PartialCorr_Ch{iChn}.npy')
np.save(save_file, corr_results)
print(f"✅ Results saved to: {save_file}")

# ===== Plot results for multiple time bins =====
Plot_figure = 1
if Plot_figure == 1:
    print("Creating plots...")
    fig, axes = plt.subplots(5, 2, figsize=(8, 20))
    axes = axes.flatten()

    # Select 10 time bins evenly distributed across all time bins
    time_bins_to_plot = np.linspace(0, N_TimeBins-1, 10, dtype=int)

    for i, time_bin in enumerate(time_bins_to_plot):
        mean_corr = np.mean(corr_results[:, :, time_bin], axis=1)
        std_corr = np.std(corr_results[:, :, time_bin], axis=1)
        lags = np.arange(len(mean_corr))
        
        axes[i].plot(lags, mean_corr, linewidth=2, color='blue', label='Partial Correlation')
        axes[i].fill_between(lags, mean_corr - std_corr, mean_corr + std_corr, alpha=0.3, color='blue')

        axes[i].set_xlabel('Time (s)') # Updated label
        axes[i].set_ylabel('Partial Correlation (r)')
        axes[i].set_title(f'Time Bin {time_bin}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

        # Set x-axis limits to -2 to 2
        axes[i].set_xlim([-2, 2])
        # Set x-axis ticks at integer steps
        axes[i].set_xticks(np.arange(-2, 2.1, 1))

        # Set y-axis limits
        y_min, y_max = np.min(mean_corr - std_corr), np.max(mean_corr + std_corr)
        y_range = y_max - y_min
        axes[i].set_ylim([y_min - 0.1*y_range, y_max + 0.1*y_range])

    plt.suptitle(f'{sub_name} Ch{iChn} auditory Partial Correlation (controlling Semantic+Visual) - All Data', fontsize=16)
    plt.tight_layout()

    plt_dir = os.path.join(save_dir,"plot")
    os.makedirs(plt_dir, exist_ok=True)
    plot_file = os.path.join(plt_dir, f'{sub_name}_PartialCorr_Ch{iChn}_AllTimeBins_AllData_read.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✅ Plot saved to: {plot_file}")
    print(f"✅ Time bins plotted: {time_bins_to_plot}")


# Calculate statistics across folds and time bins
overall_mean = np.mean(corr_results)
overall_std = np.std(corr_results)
print(f"Overall mean ± std: {overall_mean:.4f} ± {overall_std:.4f}")

print("✅ Analysis completed successfully!")
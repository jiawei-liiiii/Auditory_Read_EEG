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
N_TIME_POINTS = 820  # Renamed N_TIMES to N_TIME_POINTS
STEP = 5
N_LAG = 161
N_FOLDS = 10
RANDOM_SEED = 42


# ===== Subject Info =====
subject_index = 1001
channel_index = 14
subject_name = f'Sub{subject_index}'  

# ===== Session Info (minimal logic) =====
if subject_index == 1001:
    read_sessions = list(range(1, 8))  
elif subject_index == 1002:
    read_sessions = list(range(1, 7))
elif subject_index == 1003:
    read_sessions = list(range(1, 5))
else:
    read_sessions = list(range(1, 7))

print(f"Processing {subject_name}, Channel {channel_index}")

# ===== Load Predictability Labels =====
label_file = f'./derivatives/MetaData/expect_or_not/{subject_name}_expectation.npy'
expectation_labels = np.load(label_file)
print(f"✅ expectation_labels shape: {expectation_labels.shape}")
print(expectation_labels)


# ===== Load Neural Data =====
neural_path = f'./derivatives/eeg_processed/{subject_name}/'
neural_file = f'{subject_name}_Ch{channel_index}.npy'
neural_data = np.load(os.path.join(neural_path, neural_file))  
print("Neural data shape:", neural_data.shape)

# ===== Load Auditory Features  =====
auditory_path = f'./derivatives/Features/auditory_time_resolved/'  
auditory_file = f'{subject_name}_aud_time_res.npy' 
auditory_embedding = np.load(os.path.join(auditory_path, auditory_file)) 
print(f"✅ Auditory features shape: {auditory_embedding.shape}")

# Get number of time bins from mel data
N_TIME_BINS = auditory_embedding.shape[2]  
print(f"✅ Number of time bins: {N_TIME_BINS}")

# ========== Load Control Features ==========
# Semantic: GPT
semantic_path = f'./derivatives/Features/semantic'  
semantic_file = f'{subject_name}_semantic.npy'  
semantic_embedding = np.load(os.path.join(semantic_path, semantic_file))  
print("Semantic (GPT) shape:", semantic_embedding.shape)

# Visual: VWFA
visual_path = f'./derivatives/Features/visual'  
visual_file = f'{subject_name}_visual.npy' 
visual_embedding = np.load(os.path.join(visual_path, visual_file)) 
print("Visual (VWFA) shape:", visual_embedding.shape)

# ===== Load Exclusion Vector and Apply Mask =====
exclusion_metadata_file = f'./derivatives/MetaData/storyInd/{subject_name}_storyids.npy' 
exclusion_vector = np.load(exclusion_metadata_file)  
sample_mask = ~np.isin(exclusion_vector, [10, 20, 30])  

# Apply exclusion to all data
auditory_embedding_masked = auditory_embedding[sample_mask]  
neural_data_masked = neural_data[sample_mask]  
semantic_embedding_masked = semantic_embedding[sample_mask]  
visual_embedding_masked = visual_embedding[sample_mask] 
labels_masked = expectation_labels[sample_mask]  

print(f"✅ After masking - samples: {auditory_embedding_masked.shape[0]}")  

# ===== Equalize sample sizes for 'expected' and 'unexpected' =====
mask_expected = labels_masked == 1  
mask_unexpected = labels_masked == 0  
n_samples = min(mask_expected.sum(), mask_unexpected.sum())

print(f"✅ Expected samples: {mask_expected.sum()}, Unexpected: {mask_unexpected.sum()}")  
print(f"✅ Using {n_samples} samples for each condition")

rng = np.random.default_rng(seed=RANDOM_SEED)
idx_expected = rng.choice(np.where(mask_expected)[0], size=n_samples, replace=False)  
idx_unexpected = rng.choice(np.where(mask_unexpected)[0], size=n_samples, replace=False) 

# Prepare datasets for each condition
datasets = {
    'expected': {  
        'auditory': auditory_embedding_masked[idx_expected],  
        'semantic': semantic_embedding_masked[idx_expected],  
        'visual': visual_embedding_masked[idx_expected],  
        'neural': neural_data_masked[idx_expected]
    },
    'unexpected': {  
        'auditory': auditory_embedding_masked[idx_unexpected],
        'semantic': semantic_embedding_masked[idx_unexpected],
        'visual': visual_embedding_masked[idx_unexpected],
        'neural': neural_data_masked[idx_unexpected]
    }
}

# ===== Scale features for each condition =====
print("Scaling features...")
for condition in ['expected', 'unexpected']:  
    # Scale semantic features
    scaler_semantic = StandardScaler()
    datasets[condition]['semantic_scaled'] = scaler_semantic.fit_transform(datasets[condition]['semantic'])
    
    # Scale visual features
    scaler_visual = StandardScaler()
    datasets[condition]['visual_scaled'] = scaler_visual.fit_transform(datasets[condition]['visual'])

# ===== Initialize Result Dict =====
correlation_results = {  
    'partial_corr_expected': np.zeros((N_LAG, N_FOLDS, N_TIME_BINS)), 
    'partial_corr_unexpected': np.zeros((N_LAG, N_FOLDS, N_TIME_BINS))  
}

# ===== Main Regression Loop =====
print("Starting partial correlation analysis...")

for time_bin_index in range(N_TIME_BINS):  
    print(f"Processing time bin {time_bin_index + 1}/{N_TIME_BINS}")
    
    # Scale auditory features for current time bin for each condition
    for condition in ['expected', 'unexpected']:  
        auditory_timebin = datasets[condition]['auditory'][:, :, time_bin_index]
        scaler_auditory = StandardScaler()
        datasets[condition]['auditory_timebin_scaled'] = scaler_auditory.fit_transform(auditory_timebin)
    
    for lag_index in range(0, N_LAG): 
            
        for condition in ['expected', 'unexpected']:  
            # Get features and neural data
            X_semantic = datasets[condition]['semantic_scaled']
            X_visual = datasets[condition]['visual_scaled']
            X_auditory = datasets[condition]['auditory_timebin_scaled']
            y_neural_response = datasets[condition]['neural'][:, lag_index]  
            
            if len(y_neural_response) < N_FOLDS:
                continue
            
            # Cross-validation
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
            for fold, (train_idx, test_idx) in enumerate(kf.split(X_semantic)):
                
                # Split data
                X_S_train, X_S_test = X_semantic[train_idx], X_semantic[test_idx]
                X_V_train, X_V_test = X_visual[train_idx], X_visual[test_idx]
                X_A_train, X_A_test = X_auditory[train_idx], X_auditory[test_idx]
                y_train, y_test = y_neural_response[train_idx], y_neural_response[test_idx]
                
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
                    correlation_results[f'partial_corr_{condition}'][lag_index, fold, time_bin_index] = corr_val  
                except Exception as e:
                    # Handle cases where partial correlation fails
                    correlation_results[f'partial_corr_{condition}'][lag_index, fold, time_bin_index] = 0.0  

print("✅ Partial correlation analysis completed!")

# ===== Save Results =====
save_directory = f'./test_data/1-encoding/3-predicted/{subject_name}'  
os.makedirs(save_directory, exist_ok=True)

save_file = os.path.join(save_directory, f'{subject_name}_PartialCorr_Ch{channel_index}_timebins.npz')  
print(f"✅ Results saved to: {save_file}")

# ===== Plot results for multiple time bins =====
Plot_figure = 1
if Plot_figure == 1 :
    print("Creating plots...")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    # Select 10 time bins evenly distributed across all time bins
    time_bins_to_plot = np.linspace(0, N_TIME_BINS-1, 10, dtype=int)

    for i, time_bin in enumerate(time_bins_to_plot):
        mean_corr_expected = np.mean(correlation_results['partial_corr_expected'][:, :, time_bin], axis=1) 
        mean_corr_unexpected = np.mean(correlation_results['partial_corr_unexpected'][:, :, time_bin], axis=1) 
        lags = np.arange(len(mean_corr_expected))
        
        axes[i].plot(lags, mean_corr_expected, label='Expected', linewidth=2, color='blue') 
        axes[i].plot(lags, mean_corr_unexpected, label='Unexpected', linewidth=2, color='red') 
        axes[i].set_xlabel('Lag Index')
        axes[i].set_ylabel('Partial Correlation (r)')
        axes[i].set_title(f'Time Bin {time_bin}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Set consistent y-axis limits across all subplots
        all_vals = np.concatenate([mean_corr_expected, mean_corr_unexpected])
        y_min, y_max = np.min(all_vals), np.max(all_vals)
        y_range = y_max - y_min
        axes[i].set_ylim([y_min - 0.1*y_range, y_max + 0.1*y_range])

    plt.suptitle(f'{subject_name} Ch{channel_index} auditory Partial Correlation (controlling Semantic+Visual)', fontsize=16) 
    plt.tight_layout()

    plot_directory = './test_figures_partial/1-1_read' # Renamed plt_dir to plot_directory
    os.makedirs(plot_directory, exist_ok = True)
    plot_file = os.path.join(plot_directory, f'{subject_name}_PartialCorr_Ch{channel_index}_AllTimeBins_read.png') # Updated variable names
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✅ Plot saved to: {plot_file}")
    print(f"✅ Time bins plotted: {time_bins_to_plot}")


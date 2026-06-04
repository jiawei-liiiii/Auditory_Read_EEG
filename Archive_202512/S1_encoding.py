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

# ========== Config & Constants ===========
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
    readSessions = listenSessions = list(range(1, 8))
elif iSub == 1003:
    readSessions = list(range(1, 5))
else:
    readSessions = list(range(1, 7))



# ========== Load Exclusion Vector and Apply Mask ==========
exclusion_file = f'./derivatives/MetaData/story_epoch_match/{sub_name}_storyids.npy'
exclude_vector = np.load(exclusion_file)
mask = ~np.isin(exclude_vector, [10, 20, 30]) # remove the repeated stories

# ========== Load Neural Data ==========
neural_path = f'./derivatives/eeg_processed/{sub_name}/'
neural_file = f'{sub_name}_Ch{iChn}.npy'
neural_data = np.load(os.path.join(neural_path, neural_file))  # shape: [samples, timepoints]
print("Neural data shape:", neural_data.shape)

# ========== Load Feature Data ==========
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

# Auditory: Mel 
aud_path = f'./derivatives/Features/auditory' # Renamed path variable
aud_file = f'{sub_name}_auditory.npy'
aud_emb = np.load(os.path.join(aud_path, aud_file)) 
print("Acoustic (Mel) shape:", aud_emb.shape) 


# ========== Standardize Features ==========
scaler = StandardScaler()
sem_emb = scaler.fit_transform(sem_emb)
vis_emb = scaler.fit_transform(vis_emb)
aud_emb = scaler.fit_transform(aud_emb)

# Concatenate all features
X_full = np.hstack([sem_emb, vis_emb, aud_emb])  # shape: [samples, 150]
X = X_full[mask]

print(f"Applied mask: excluded {np.sum(~mask)} samples, kept {np.sum(mask)} samples")

# ========== Cross-validation ==========
kf = KFold(n_splits=N_FOLDS)

# ========== Initialize Results ==========
corr_results = {key: np.zeros((N_LAG, N_FOLDS)) for key in [
    'corr_full', 'partial_corr_S', 'partial_corr_A', 'partial_corr_V',
    'corr_V_A', 'corr_S_A', 'corr_S_V', 'corr_V', 'corr_S', 'corr_A'
]}

# ========== Main Loop ==========
for iLag in range(0, N_LAG):
    print(f"Time lag {iLag}")
    y_all = neural_data[:, iLag]
    y_target = y_all[mask]

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_target[train_idx], y_target[test_idx]

        model_full = LinearRegression().fit(X_train, y_train)
        y_pred_full = model_full.predict(X_test)
        corr_results['corr_full'][iLag, fold] = pearsonr(y_test, y_pred_full)[0]

        # Define feature slices
        slices = {
            'S': slice(0, EMBED_DIM),
            'V': slice(EMBED_DIM, 2 * EMBED_DIM),
            'A': slice(2 * EMBED_DIM, 3 * EMBED_DIM),
            'S_V': np.r_[:2 * EMBED_DIM],
            'S_A': np.r_[0:EMBED_DIM, 2 * EMBED_DIM:3 * EMBED_DIM],
            'V_A': np.r_[EMBED_DIM:3 * EMBED_DIM]
        }

        # Individual and combined models
        preds = {}
        for key, sl in slices.items():
            model = LinearRegression().fit(X_train[:, sl], y_train)
            preds[key] = model.predict(X_test[:, sl])
            corr_results[f'corr_{key}'][iLag, fold] = pearsonr(y_test, preds[key])[0]

        # Partial correlation
        df = pd.DataFrame({
            'y_test': y_test,
            'y_pred_full': y_pred_full,
            **{f'y_pred_{k}': v for k, v in preds.items()}
        })

        for target, covar in [('S', 'V_A'), ('V', 'S_A'), ('A', 'S_V')]:
            key = f'partial_corr_{target}'
            corr_val = partial_corr(df, x='y_test', y=f'y_pred_{target}', covar=f'y_pred_{covar}')['r'].values[0]
            corr_results[key][iLag, fold] = corr_val

# ========== Save ==========
save_dir = f'./test_data/1-encoding/{sub_name}'
os.makedirs(save_dir, exist_ok=True)
out_path = os.path.join(save_dir, f'{sub_name}_Combine_Ch{iChn}_corr.npz')
np.savez(out_path, **corr_results)
print("Results saved to:", out_path)

# ========== Plot ==========
Plot_figure = 1
if Plot_figure == 1:
    # Time vector from -2 to 2 s with 0.025 s step
    time_points = np.arange(-2, 2 + 0.025, 0.025)

    # Plot correlations
    plt.figure(figsize=(10, 6))
    for key, arr in corr_results.items():
        plt.plot(time_points, arr[:, 0], label=key)

    plt.xlabel('Time (s)')
    plt.ylabel('Correlation r')
    plt.title(f'Sub{iSub} Ch{iChn} Correlations')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'Sub{iSub}_Ch{iChn}_correlations_Read.png'), dpi=300)
    plt.close()
    print(f"Plot saved as Sub{iSub}_Ch{iChn}_correlations_Read.png in current folder")
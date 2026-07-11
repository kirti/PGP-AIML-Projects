"""
Synthetic ReneWind sensor data generator.
Mimics schema/shape of the real Train.csv / Test.csv:
 - Train: 20,000 rows x 41 cols (V1-V40 + Target)
 - Test:   5,000 rows x 41 cols
 - Missing values only in V1 and V2 (a small handful, <0.1%)
 - Target highly imbalanced (minority class ~ "failure")
 - Features are anonymized/ciphered sensor readings -> modeled as
   roughly normal-ish continuous variables with realistic min/max ranges
NOTE: this is fully synthetic and carries no real sensor/business data.
"""
import numpy as np
import pandas as pd

def make_dataset(n_rows, seed, minority_frac=0.056):
    rng = np.random.default_rng(seed)
    n_features = 40

    # Target: imbalanced binary (1 = failure, minority class)
    target = rng.choice([0, 1], size=n_rows, p=[1 - minority_frac, minority_frac])

    data = {}
    for i in range(1, n_features + 1):
        # base noise
        base = rng.normal(loc=0.0, scale=3.0, size=n_rows)
        # small class-conditional shift so models have *some* signal to find,
        # mimicking that certain sensor channels correlate with failure
        if i in (3, 7, 15, 22, 31):  # a handful of "informative" features
            shift = rng.normal(loc=1.2, scale=0.3)
            base = base + target * shift
        data[f"V{i}"] = base

    df = pd.DataFrame(data)
    # Clip extreme tails to roughly match real observed ranges (~-12 to ~15)
    df = df.clip(lower=-12, upper=15.5)
    df["Target"] = target
    return df

train = make_dataset(20000, seed=42, minority_frac=0.056)
test = make_dataset(5000, seed=99, minority_frac=0.056)

# Introduce missing values only in V1 and V2, matching real pattern (~18 in train, ~5-6 in test)
rng = np.random.default_rng(1)
train_missing_idx_v1 = rng.choice(train.index, size=18, replace=False)
train_missing_idx_v2 = rng.choice(train.index, size=18, replace=False)
train.loc[train_missing_idx_v1, "V1"] = np.nan
train.loc[train_missing_idx_v2, "V2"] = np.nan

rng2 = np.random.default_rng(2)
test_missing_idx_v1 = rng2.choice(test.index, size=5, replace=False)
test_missing_idx_v2 = rng2.choice(test.index, size=6, replace=False)
test.loc[test_missing_idx_v1, "V1"] = np.nan
test.loc[test_missing_idx_v2, "V2"] = np.nan

train.to_csv('/mnt/user-data/outputs/Train_DUMMY.csv', index=False)
test.to_csv('/mnt/user-data/outputs/Test_DUMMY.csv', index=False)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Train target distribution:\n", train['Target'].value_counts(normalize=True))
print("Test target distribution:\n", test['Target'].value_counts(normalize=True))
print("Train missing:\n", train.isnull().sum()[train.isnull().sum() > 0])
print("Test missing:\n", test.isnull().sum()[test.isnull().sum() > 0])

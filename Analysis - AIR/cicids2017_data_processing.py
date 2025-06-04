import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

samples_per_class = 30000

path = os.path.join('/mnt/AI-DATA/alara/CiberIA_O1/Data')

print("Reading data...")
print(f"Sampling per class: {samples_per_class} Total samples: {samples_per_class * 7}")
df = pd.read_csv(os.path.join(path, 'CIC-IDS2017.csv'))

# remove 'Infiltration' and 'Heartbleed' classes
df = df[~df['Attack Type'].isin(['Infiltration', 'Heartbleed'])]
print("Data shape:", df.shape)
print("Data columns:", df.columns)

print("Encoding attack types...")
le = LabelEncoder()
df['Attack Number'] = le.fit_transform(df['Attack Type'])
print("Encoded attack classes:")
for val in sorted(df['Attack Number'].unique()):
    print(f"{val}: {le.inverse_transform([val])[0]}")

print("Number of samples of each attack type:")
print(df['Attack Type'].value_counts())

print("Reducing memory usage...")
old_memory_usage = df.memory_usage().sum() / 1024 ** 2
print(f'Initial memory usage: {old_memory_usage:.2f} MB')
for col in df.columns:
    col_type = df[col].dtype
    if col_type != object:
        c_min = df[col].min()
        c_max = df[col].max()
        if str(col_type).find('float') >= 0 and c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
            df[col] = df[col].astype(np.float32)
        elif str(col_type).find('int') >= 0 and c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)
new_memory_usage = df.memory_usage().sum() / 1024 ** 2
print(f"Final memory usage: {new_memory_usage:.2f} MB")
print(f'Reduced memory usage: {1 - (new_memory_usage / old_memory_usage):.2%}')

data = df.copy()

attacks = data['Attack Type']
data = data.select_dtypes(include=['float32', 'int32'])

print("Standardizing features...")
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data)

print("Applying PCA...")
size = len(data.columns) // 2
pca = PCA(n_components=size)
transformed_features = pca.fit_transform(scaled_features)
print(f'Information retained: {sum(pca.explained_variance_ratio_):.2%}')
new_data_pca = pd.DataFrame(transformed_features, columns=[f'PC{i+1}' for i in range(size)])
new_data_pca['Attack Type'] = attacks.values

print("Selecting K-best features...")
k = len(data.columns) // 2
selector = SelectKBest(f_classif, k=k)
selector.fit(data, attacks)
selected_features_32 = data.columns[selector.get_support()]
print(f'Selected Features: {selected_features_32}')
new_data_k_32 = data[selected_features_32]
new_data_k_32['Attack Type'] = attacks.values

print("Applying SMOTE to balance data...")
def balance_and_split(dataset):
    class_counts = dataset['Attack Type'].value_counts()
    selected_classes = class_counts[class_counts > 1950]
    class_names = selected_classes.index
    selected = dataset[dataset['Attack Type'].isin(class_names)]
    
    dfs = []
    for name in class_names:
        df_class = selected[selected['Attack Type'] == name]
        if len(df_class) > 1950:
            df_class = df_class.sample(n=samples_per_class, replace=True, random_state=0)
        dfs.append(df_class)
    
    balanced_df = pd.concat(dfs, ignore_index=True)
    X = balanced_df.drop('Attack Type', axis=1)
    y = balanced_df['Attack Type']
    smote = SMOTE(sampling_strategy='auto', random_state=0)
    X_upsampled, y_upsampled = smote.fit_resample(X, y)
    balanced_data = pd.DataFrame(X_upsampled)
    balanced_data['Attack Type'] = y_upsampled
    balanced_data = balanced_data.sample(frac=1)
    return train_test_split(balanced_data.drop('Attack Type', axis=1), balanced_data['Attack Type'], test_size=0.2, random_state=0)

X_train_S_PCA, X_test_S_PCA, y_train_S_PCA, y_test_S_PCA = balance_and_split(new_data_pca)
X_train_S_32, X_test_S_32, y_train_S_32, y_test_S_32 = balance_and_split(new_data_k_32)

# no_smote
# Updated sampling: Get 35000 samples while ensuring that every class with >1950 samples
# (excluding 'Infiltration' and 'Heartbleed') contributes at least 10 samples.

# Extract features and labels from new_data_pca
features = new_data_pca.drop('Attack Type', axis=1)
labels = new_data_pca['Attack Type']

# Calculate full class counts in new_data_pca
full_counts = new_data_pca['Attack Type'].value_counts()

# Identify classes to stratify (classes with >1950 samples, excluding 'Infiltration' and 'Heartbleed')
classes_to_stratify = [cls for cls, count in full_counts.items() if count > 1950 and cls not in ['Infiltration', 'Heartbleed']]

# For each of these classes, take at least 10 random samples
stratified_samples = []
for cls in classes_to_stratify:
    cls_data = new_data_pca[new_data_pca['Attack Type'] == cls]
    stratified_samples.append(cls_data.sample(n=10, random_state=0))

# Combine the stratified samples
stratified_df = pd.concat(stratified_samples, ignore_index=True)

# Remove the stratified rows from the original DataFrame to avoid duplication
remaining_df = new_data_pca.drop(stratified_df.index)

# Determine how many more samples to draw to reach a total of 35000 samples
remaining_count = (samples_per_class * 7) - stratified_df.shape[0]

# Randomly sample the remaining rows
remaining_sample = remaining_df.sample(n=remaining_count, random_state=0)

# Combine stratified and remaining samples and shuffle the dataframe
sampled_df = pd.concat([stratified_df, remaining_sample]).sample(frac=1, random_state=0)

# Perform a train/test split on the final sampled dataset
X_train_N_PCA, X_test_N_PCA, y_train_N_PCA, y_test_N_PCA = train_test_split(
    sampled_df.drop('Attack Type', axis=1),
    sampled_df['Attack Type'],
    test_size=0.2,
    random_state=0
)

# no_smote for new_data_k_32

# Extract features and labels from new_data_k_32
features = new_data_k_32.drop('Attack Type', axis=1)
labels = new_data_k_32['Attack Type']

# Calculate full class counts in new_data_k_32
full_counts = new_data_k_32['Attack Type'].value_counts()

# Identify classes to stratify (classes with >1950 samples, excluding 'Infiltration' and 'Heartbleed')
classes_to_stratify = [cls for cls, count in full_counts.items() if count > 1950 and cls not in ['Infiltration', 'Heartbleed']]

# For each of these classes, take at least 10 random samples
stratified_samples = []
for cls in classes_to_stratify:
    cls_data = new_data_k_32[new_data_k_32['Attack Type'] == cls]
    stratified_samples.append(cls_data.sample(n=10, random_state=0))

# Combine the stratified samples
stratified_df = pd.concat(stratified_samples, ignore_index=True)

# Remove the stratified rows from the original DataFrame to avoid duplication
remaining_df = new_data_k_32.drop(stratified_df.index)

# Determine how many more samples to draw to reach a total of 35000 samples
remaining_count = (samples_per_class * 7) - stratified_df.shape[0]

# Randomly sample the remaining rows
remaining_sample = remaining_df.sample(n=remaining_count, random_state=0)

# Combine stratified and remaining samples and shuffle the dataframe
sampled_df = pd.concat([stratified_df, remaining_sample]).sample(frac=1, random_state=0)

# Perform a train/test split on the final sampled dataset
X_train_N_K, X_test_N_K, y_train_N_K, y_test_N_K = train_test_split(
    sampled_df.drop('Attack Type', axis=1),
    sampled_df['Attack Type'],
    test_size=0.2,
    random_state=0
)

print("Data preprocessing complete.")

print(f"X_train_S_PCA: {X_train_S_PCA.shape}, y_train_S_PCA: {y_train_S_PCA.shape}, num_classes: {len(y_train_S_PCA.unique())}")
print(f"X_test_S_PCA: {X_test_S_PCA.shape}, y_test_S_PCA: {y_test_S_PCA.shape}, num_classes: {len(y_test_S_PCA.unique())}")
print(f"X_train_S_32: {X_train_S_32.shape}, y_train_S_32: {y_train_S_32.shape}, num_classes: {len(y_train_S_32.unique())}")
print(f"X_test_S_32: {X_test_S_32.shape}, y_test_S_32: {y_test_S_32.shape}, num_classes: {len(y_test_S_32.unique())}")
print(f"X_train_N_PCA: {X_train_N_PCA.shape}, y_train_N_PCA: {y_train_N_PCA.shape}, num_classes: {len(y_train_N_PCA.unique())}")
print(f"X_test_N_PCA: {X_test_N_PCA.shape}, y_test_N_PCA: {y_test_N_PCA.shape}, num_classes: {len(y_test_N_PCA.unique())}")
print(f"X_train_N_K: {X_train_N_K.shape}, y_train_N_K: {y_train_N_K.shape}, num_classes: {len(y_train_N_K.unique())}")
print(f"X_test_N_K: {X_test_N_K.shape}, y_test_N_K: {y_test_N_K.shape}, num_classes: {len(y_test_N_K.unique())}")

# get number of classes represented in each dataset
print(f"Classes in X_train_S_PCA: {y_train_S_PCA.unique()}")
print(f"Classes in X_test_S_PCA: {y_test_S_PCA.unique()}")
print(f"Classes in X_train_S_32: {y_train_S_32.unique()}")
print(f"Classes in X_test_S_32: {y_test_S_32.unique()}")
print(f"Classes in X_train_N_PCA: {y_train_N_PCA.unique()}")
print(f"Classes in X_test_N_PCA: {y_test_N_PCA.unique()}")
print(f"Classes in X_train_N_K: {y_train_N_K.unique()}")
print(f"Classes in X_test_N_K: {y_test_N_K.unique()}")


# save the variables
print("Saving variables...")
np.savez_compressed(os.path.join(path, 'Data_CIC_IDS_2017.npz'),
                    X_train_S_PCA=X_train_S_PCA, X_test_S_PCA=X_test_S_PCA, y_train_S_PCA=y_train_S_PCA, y_test_S_PCA=y_test_S_PCA,
                    X_train_S_K=X_train_S_32, X_test_S_K=X_test_S_32, y_train_S_K=y_train_S_32, y_test_S_K=y_test_S_32,
                    X_train_N_PCA=X_train_N_PCA, X_test_N_PCA=X_test_N_PCA, y_train_N_PCA=y_train_N_PCA, y_test_N_PCA=y_test_N_PCA,
                    X_train_N_K=X_train_N_K, X_test_N_K=X_test_N_K, y_train_N_K=y_train_N_K, y_test_N_K=y_test_N_K)

print("Variables saved. Filesize: ", os.path.getsize(os.path.join(path, 'Data_CIC_IDS_2017.npz')) / 1024 ** 2, "MB")	

print("Data preprocessing complete.")

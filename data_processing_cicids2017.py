import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

path = os.path.join(os.getcwd(), 'Data')

print("Reading data...")
df = pd.read_csv(os.path.join(path, 'CIC-IDS2017.csv'))

print("Checking for duplicate values...")
duplicates = df.duplicated()
print(f'There are {duplicates.sum()} duplicated rows in the dataset. Total shape: {df.shape}')
df.drop_duplicates(inplace=True)
print(f'Duplicated rows removed. New shape: {df.shape}')

print("Checking for missing and infinite values...")
numeric_cols = df.select_dtypes(include=np.number).columns
inf_count = np.isinf(df[numeric_cols]).sum()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
missing = df.isna().sum()

print("Removing missing values...")
df.dropna(inplace=True)

print("Mapping attack labels...")
attack_map = {
    'BENIGN': 'BENIGN',
    'DDoS': 'DDoS',
    'DoS Hulk': 'DoS',
    'DoS GoldenEye': 'DoS',
    'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS',
    'PortScan': 'Port Scan',
    'FTP-Patator': 'Brute Force',
    'SSH-Patator': 'Brute Force',
    'Bot': 'Bot',
    'Web Attack Â Brute Force': 'Web Attack',
    'Web Attack Â XSS': 'Web Attack',
    'Web Attack Â Sql Injection': 'Web Attack',
    'Infiltration': 'Infiltration',
    'Heartbleed': 'Heartbleed'
}
df['Attack Type'] = df['Label'].map(attack_map)
df.drop('Label', axis=1, inplace=True)

print("Encoding attack types...")
le = LabelEncoder()
df['Attack Number'] = le.fit_transform(df['Attack Type'])
print("Encoded attack classes:")
for val in sorted(df['Attack Number'].unique()):
    print(f"{val}: {le.inverse_transform([val])[0]}")

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

print("Dropping columns with one unique value...")
num_unique = df.nunique()
one_variable = num_unique[num_unique == 1]
not_one_variable = num_unique[num_unique > 1].index
data = df[not_one_variable]

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

print("Selecting K-best features (k=32)...")
k = 32
selector = SelectKBest(f_classif, k=k)
selector.fit(data, attacks)
selected_features_32 = data.columns[selector.get_support()]
print(f'Selected Features (k=20): {selected_features_32}')
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
        if len(df_class) > 2500:
            df_class = df_class.sample(n=150000, replace=True, random_state=0)
        dfs.append(df_class)
    
    balanced_df = pd.concat(dfs, ignore_index=True)
    X = balanced_df.drop('Attack Type', axis=1)
    y = balanced_df['Attack Type']
    smote = SMOTE(sampling_strategy='auto', random_state=0)
    X_upsampled, y_upsampled = smote.fit_resample(X, y)
    balanced_data = pd.DataFrame(X_upsampled)
    balanced_data['Attack Type'] = y_upsampled
    balanced_data = balanced_data.sample(frac=1)
    return train_test_split(balanced_data.drop('Attack Type', axis=1), balanced_data['Attack Type'], test_size=0.3, random_state=0)

X_train_S_PCA, X_test_S_PCA, y_train_S_PCA, y_test_S_PCA = balance_and_split(new_data_pca)
X_train_S_32, X_test_S_32, y_train_S_32, y_test_S_32 = balance_and_split(new_data_k_32)

# no_smote
features = new_data_pca.drop('Attack Type', axis=1)
labels = new_data_pca['Attack Type']
features = features.sample(n=1050000, random_state=0)
labels = labels.sample(n=1050000, random_state=0)
X_train_N_PCA, X_test_N_PCA, y_train_N_PCA, y_test_N_PCA = train_test_split(features, labels, test_size=0.3, random_state=0)

# no_smote
features = new_data_k_32.drop('Attack Type', axis=1)
labels = new_data_k_32['Attack Type']
features = features.sample(n=1050000, random_state=0)
labels = labels.sample(n=1050000, random_state=0)
X_train_N_32, X_test_N_32, y_train_N_32, y_test_N_32 = train_test_split(features, labels, test_size=0.3, random_state=0)

print("Data preprocessing complete.")

# save the variables
print("Saving variables...")
np.savez_compressed(os.path.join(path, 'Data_CIC_IDS_2017.npz'),
                    X_train_S_PCA=X_train_S_PCA, X_test_S_PCA=X_test_S_PCA, y_train_S_PCA=y_train_S_PCA, y_test_S_PCA=y_test_S_PCA,
                    X_train_S_32=X_train_S_32, X_test_S_32=X_test_S_32, y_train_S_32=y_train_S_32, y_test_S_32=y_test_S_32,
                    X_train_N_PCA=X_train_N_PCA, X_test_N_PCA=X_test_N_PCA, y_train_N_PCA=y_train_N_PCA, y_test_N_PCA=y_test_N_PCA,
                    X_train_N_32=X_train_N_32, X_test_N_32=X_test_N_32, y_train_N_32=y_train_N_32, y_test_N_32=y_test_N_32)

print("Variables saved.")

print("Data preprocessing complete.")

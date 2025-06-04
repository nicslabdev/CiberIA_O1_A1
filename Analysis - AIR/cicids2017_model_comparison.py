import os
import warnings
import numpy as np
import pandas as pd
from time import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgbm
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Reshape
from sklearn.svm import LinearSVC

def create_nn_model(input_shape, num_classes):
    model = Sequential([
        # Reshape the input vector to a sequence with one channel
        Reshape((input_shape, 1), input_shape=(input_shape,)),
        
        # Convolutional layers
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # Flatten and Dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test, preprocessing, is_neural=False):
    start_time = time()
    
    # Create label encoder for models that require numeric labels
    le = LabelEncoder()
    if isinstance(model, (xgb.XGBClassifier, SVC)):
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)
    else:
        y_train_encoded = y_train
        y_test_encoded = y_test
    
    if is_neural:
        y_train_cat = to_categorical(pd.Categorical(y_train).codes)
        y_test_cat = to_categorical(pd.Categorical(y_test).codes)
        
        model.fit(X_train, y_train_cat,
                 epochs=10,
                 batch_size=32,
                 validation_split=0.2,
                 verbose=0)
        
        y_pred_prob = model.predict(X_test)
        y_pred = np.unique(y_test)[np.argmax(y_pred_prob, axis=1)]
    else:
        model.fit(X_train, y_train_encoded)
        y_pred_encoded = model.predict(X_test)
        
        # Convert predictions back to original labels if needed
        if isinstance(model, (xgb.XGBClassifier, SVC)):
            y_pred = le.inverse_transform(y_pred_encoded)
        else:
            y_pred = y_pred_encoded
    
    exec_time = time() - start_time
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    # get classification report and confusion matrix and save to a file
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f'/mnt/AI-DATA/alara/CiberIA_O1/Data/CIC-IDS2017_{preprocessing}_{model.__class__.__name__}_classification_report.csv')

    # Save confusion matrix as table
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test))
    cm_df.to_csv(f'/mnt/AI-DATA/alara/CiberIA_O1/Data/CIC-IDS2017_{preprocessing}_{model.__class__.__name__}_confusion_matrix.csv')

    # Save confusion matrix plot
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='magma', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f'Confusion Matrix for {model.__class__.__name__}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'/mnt/AI-DATA/alara/CiberIA_O1/Data/CIC-IDS2017_{preprocessing}_{model.__class__.__name__}_confusion_matrix.png', dpi=500)

    print(f"Model: {model.__class__.__name__}, Preprocessing: {preprocessing}")
    print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Execution Time: {exec_time:.2f}s")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(cm)
    print(cm_df)
    plt.close()
    
    return accuracy, f1, exec_time

def create_results_table(X_train, X_test, y_train, y_test, preprocessing):
    results = []
    
    # 1. Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=85)
    rf_acc, rf_f1, rf_time = evaluate_model(rf_model, X_train, X_test, y_train, y_test, preprocessing)
    results.append(('RF', rf_acc, rf_f1, rf_time))
    
    # 2. k-NN
    print("Training k-NN...")
    knn_model = KNeighborsClassifier(n_neighbors=len(np.unique(y_train)), n_jobs=-1)
    knn_acc, knn_f1, knn_time = evaluate_model(knn_model, X_train, X_test, y_train, y_test, preprocessing)
    results.append(('k-NN', knn_acc, knn_f1, knn_time))
    
    # 3. LGBM
    print("Training LGBM...")
    lgbm_model = lgbm.LGBMClassifier(n_estimators=100, random_state=85, verbosity=-1)
    lgbm_acc, lgbm_f1, lgbm_time = evaluate_model(lgbm_model, X_train, X_test, y_train, y_test, preprocessing)
    results.append(('LGBM', lgbm_acc, lgbm_f1, lgbm_time))
        
    # 4. Neural Network
    print("Training Neural Network...")
    nn_model = create_nn_model(X_train.shape[1], len(np.unique(y_train)))
    nn_acc, nn_f1, nn_time = evaluate_model(nn_model, X_train, X_test, y_train, y_test,preprocessing, is_neural=True)
    results.append(('NN', nn_acc, nn_f1, nn_time))
    
    # 5. XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=85)
    xgb_acc, xgb_f1, xgb_time = evaluate_model(xgb_model, X_train, X_test, y_train, y_test, preprocessing)
    results.append(('XGBoost', xgb_acc, xgb_f1, xgb_time))
    
    # 6. SVM
    print("Training Linear SVM...")
    svm_model = LinearSVC(random_state=85, dual=False, max_iter=10000)
    svm_acc, svm_f1, svm_time = evaluate_model(svm_model, X_train, X_test, y_train, y_test, preprocessing)
    results.append(('SVM', svm_acc, svm_f1, svm_time))
    
    # 7. LR[RF+LGBM]
    print("Training LR[RF+LGBM]...")
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=85)),
        ('lgbm', lgbm.LGBMClassifier(n_estimators=100, random_state=85, verbosity=-1))
    ]
    lr_stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5
    )
    lr_acc, lr_f1, lr_time = evaluate_model(lr_stack, X_train, X_test, y_train, y_test, preprocessing)
    results.append(('LR[RF+LGBM]', lr_acc, lr_f1, lr_time))
    
    # Create DataFrame
    df = pd.DataFrame(results, columns=['Model', 'Acc', 'F1', 'Time (s)'])
    df.set_index('Model', inplace=True)
    
    return df

def main():
    print("Loading CIC-IDS2017 data...")
    path = os.path.join('/mnt/AI-DATA/alara/CiberIA_O1/Data')
    print("Reading data...")
    data = np.load(os.path.join(path, 'Data_CIC_IDS_2017.npz'), allow_pickle=True)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # NO-SMOTE + PCA
    print("\nEvaluating NO-SMOTE + PCA...")
    results_n_pca = create_results_table(
        data['X_train_N_PCA'], data['X_test_N_PCA'],
        data['y_train_N_PCA'], data['y_test_N_PCA'],
        'NO-SMOTE_PCA'
    )
    
    # NO-SMOTE + Top 32
    print("\nEvaluating NO-SMOTE + Top 32...")
    results_n_K = create_results_table(
        data['X_train_N_K'], data['X_test_N_K'],
        data['y_train_N_K'], data['y_test_N_K'],
        'NO-SMOTE_TopK'
    )
    
    # SMOTE + PCA
    print("\nEvaluating SMOTE + PCA...")
    results_s_pca = create_results_table(
        data['X_train_S_PCA'], data['X_test_S_PCA'],
        data['y_train_S_PCA'], data['y_test_S_PCA'],
        'SMOTE_PCA'
    )
    
    # SMOTE + Top 32
    print("\nEvaluating SMOTE + Top 32...")
    results_s_K = create_results_table(
        data['X_train_S_K'], data['X_test_S_K'],
        data['y_train_S_K'], data['y_test_S_K'],
        'SMOTE_TopK'
    )
    
    # Combine all results into a single table
    all_results = pd.concat([
        results_n_pca.add_suffix('_NO-SMOTE_PCA'),
        results_n_K.add_suffix('_NO-SMOTE_TopK'),
        results_s_pca.add_suffix('_SMOTE_PCA'),
        results_s_K.add_suffix('_SMOTE_TopK')
    ], axis=1)
    
    # Save combined results
    all_results.to_csv('/mnt/AI-DATA/alara/CiberIA_O1/Data/Results_CIC_IDS_2017.csv')
    print("\nAll results saved to CSV files")

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'    
    main() 
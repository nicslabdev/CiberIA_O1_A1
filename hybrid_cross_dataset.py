import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

# Create images directory if it doesn't exist
os.makedirs('images', exist_ok=True)

def create_binary_model(input_shape):
    model = Sequential([
        Dense(128, input_dim=input_shape, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    return model

def plot_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join('images', f'hybrid_cross_{title.lower().replace(" ", "_")}.png'))
    plt.close()

def main():
    print("Loading datasets...")
    path = os.path.join(os.getcwd(), 'Data')
    
    # Load CIC-IDS2017 (training dataset)
    cic_data = np.load(os.path.join(path, 'Data_CIC_IDS_2017.npz'), allow_pickle=True)
    X_train_original = cic_data['X_train_S_PCA']
    y_train = cic_data['y_train_S_PCA']
    
    # Load UNSW-NB15 (testing dataset)
    unsw_data = np.load(os.path.join(path, 'Data_UNSW_NB15.npz'), allow_pickle=True)
    X_test_original = unsw_data['X_test_S_PCA']
    y_test = unsw_data['y_test_S_PCA']

    print("Original shapes:")
    print("Training set shape:", X_train_original.shape)
    print("Test set shape:", X_test_original.shape)

    # Match dimensions by taking the first n_features
    print("\nMatching dimensions...")
    n_features = min(X_train_original.shape[1], X_test_original.shape[1])
    X_train = X_train_original[:, :n_features]
    X_test = X_test_original
    
    print("After dimension matching:")
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)

    # Layer 1: Binary Classification (Normal vs Attack)
    print("\nLayer 1: Binary Classification...")
    
    # Convert labels to binary (Normal vs Attack)
    y_train_binary = np.where(y_train == 'BENIGN', 0, 1)
    y_test_binary = np.where(y_test == 'BENIGN', 0, 1)

    # Convert to categorical for neural network
    y_train_cat = to_categorical(y_train_binary)
    y_test_cat = to_categorical(y_test_binary)

    # Train binary classifier
    binary_model = create_binary_model(X_train.shape[1])
    binary_history = binary_model.fit(X_train, y_train_cat,
                                    epochs=10,
                                    batch_size=256,
                                    validation_split=0.2,
                                    verbose=1)

    # Evaluate binary classifier
    binary_pred_prob = binary_model.predict(X_test)
    binary_pred = np.argmax(binary_pred_prob, axis=1)
    binary_accuracy = accuracy_score(y_test_binary, binary_pred)
    print("\nBinary Classification Results:")
    print(f"Accuracy: {binary_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_binary, binary_pred))
    plot_confusion_matrix(y_test_binary, binary_pred, ['Normal', 'Attack'], "Binary")

    # Layer 2: Multi-class Classification (Attack Types)
    print("\nLayer 2: Multi-class Classification...")
    
    # Filter attack samples for training
    X_train_attacks = X_train[y_train_binary == 1]
    y_train_attacks = y_train[y_train_binary == 1]
    
    # Filter attack samples for testing
    X_test_attacks = X_test[binary_pred == 1]
    y_test_attacks = y_test[binary_pred == 1]

    print("Attack samples - Training:", X_train_attacks.shape)
    print("Attack samples - Testing:", X_test_attacks.shape)

    # Create and train stacking classifier
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
        ('lgb', lgb.LGBMClassifier(n_estimators=100, random_state=42))
    ]
    
    stack_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5
    )

    # Train stacking classifier
    stack_clf.fit(X_train_attacks, y_train_attacks)
    
    # Make predictions
    y_pred_attacks = stack_clf.predict(X_test_attacks)
    attack_accuracy = accuracy_score(y_test_attacks, y_pred_attacks)
    
    print("\nMulti-class Classification Results:")
    print(f"Accuracy: {attack_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_attacks, y_pred_attacks))
    plot_confusion_matrix(y_test_attacks, y_pred_attacks, 
                         np.unique(y_test_attacks), "Multiclass")

    # Calculate overall accuracy
    # First, get predictions for all samples
    y_pred_all = np.array(['BENIGN'] * len(y_test))  # Initialize all as BENIGN
    attack_indices = np.where(binary_pred == 1)[0]  # Get indices of predicted attacks
    y_pred_all[attack_indices] = y_pred_attacks  # Replace with attack predictions
    
    overall_accuracy = accuracy_score(y_test, y_pred_all)
    print("\nOverall Results:")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print("\nOverall Classification Report:")
    print(classification_report(y_test, y_pred_all))
    plot_confusion_matrix(y_test, y_pred_all, np.unique(y_test), "Overall")

    # Print final results
    print("\nFinal Results Summary:")
    print(f'Binary Classification Accuracy: {binary_accuracy:.4f}')
    print(f'Attack Classification Accuracy: {attack_accuracy:.4f}')
    print(f'Overall System Accuracy: {overall_accuracy:.4f}')

if __name__ == "__main__":
    main() 
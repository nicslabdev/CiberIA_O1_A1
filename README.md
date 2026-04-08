# CiberIA_O1_A1 🛡️🤖
![Fondos_INCIBE](https://github.com/nicslabdev/CiberIA_O1_A1/raw/main/logo_fondos_incibe.png)
This repository is part of the project "CiberIA: Investigación e Innovación para la Integración de Ciberseguridad e Inteligencia Artificial" (Proyecto C079/23), financed by "European Union NextGeneration-EU, the Recovery Plan, Transformation and Resilience", through INCIBE.

## ⚡ Project Overview
CiberIA_O1_A1 is the first assignment (Actividad 1) of the CiberIA project. The goal is to analyze and compare the performance of various machine learning models on multiple intrusion detection datasets as previous work, and design a maintainable and verifiable AI detection system. This project covers data processing, feature engineering, model training, evaluation, and a unified threat detection framework.

## 🎯 Objectives
- Load and preprocess raw network intrusion datasets (CIC-IDS2017, CIC-IDS2018, CIC-IDS2019, UNSW-NB15).
- Apply dimensionality reduction (PCA) and feature selection (Top-K) techniques.
- Train and evaluate different classification algorithms (KNN, Random Forest, Linear SVC, LGBM, XGBoost, Stacking, Sequential neural networks).
- Compare results across datasets and sampling strategies (no-SMOTE vs. SMOTE).
- Develop a modular threat detection system (framework) to integrate data processing and inference.

## 🚀 Stats at a Glance 📊
| Metric           | Value                                            |
|------------------|--------------------------------------------------|
| 📁 Datasets      | 4 (CIC-IDS2017, 2018, 2019, UNSW-NB15)           |
| 🧩 Features      | PCA & Top-K                                      |
| 🤖 Models        | KNN, Random Forest, Linear SVC, LGBM, XGBoost, Stacking, Neural Network |
| 🔄 Sampling      | No-SMOTE & SMOTE                                 |
| 🧪 Experiments    | 112 (4 datasets × 7 models × 2 strategies × 2 techniques) |

## 🗂️ Repository Structure
```
CiberIA_O1_A1/
├── Analysis - AIR/       # Scripts and Jupyter notebooks for analysis and visualization
├── Data/                # Raw CSVs, processed NPZ files, classification reports, confusion matrices, and result summaries
├── Framework/           # Detection framework modules, example notebook, and requirements for deployment
├── .gitignore
└── README.md            # Project overview and setup instructions
```

### Analysis - AIR
Contains Python scripts and Jupyter notebooks that:
- Load processed data (NPZ files)
- Train classifiers with/without SMOTE and Top-K feature selection
- Generate heatmaps, confusion matrices, and performance metrics
- Compare models across datasets

### Data
- **Raw CSVs**: Original datasets (CIC-DDoS2019.csv, CIC-IDS2017.csv, etc.)
- **Processed NPZ**: `Data_CIC_IDS_*.npz`, `Data_UNSW_NB15.npz`
- **Reports & Matrices**: CSV and PNG files with classification reports and confusion matrices for each model variation
- **Results CSVs**: Tabular summaries (`Results_CIC_IDS_*.csv`, `Results_UNSW_NB15.csv`)
- **Auxiliary**: `info_gain_combined.csv`, `model_dict.pkl`

### Framework
- **modules/**: Reusable Python modules for data loading, preprocessing, and evaluation
- **threat_detection_system.py**: Entry-point script to run end-to-end detection pipeline
- **cicids2018_Notebook.ipynb**: Example notebook demonstrating framework usage on CIC-IDS2018
- **requirements.txt**: Dependencies for the detection framework

## 🛠️ Setup Instructions
1. Create and activate a virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```
2. Install dependencies for data analysis and visualization:
   ```bash
   pip install -r "Analysis - AIR/requirements.txt"
   ```
3. Install framework dependencies:
   ```bash
   pip install -r Framework/requirements.txt
   ```

## 📝 License
This repository is part of the project "CiberIA: Investigación e Innovación para la Integración de Ciberseguridad e Inteligencia Artificial" (Proyecto C079/23), financed by "European Union NextGeneration-EU, the Recovery Plan, Transformation and Resilience", through INCIBE. It has also been partially supported by the project SecAI (PID2022-139268OB-I00) funded by the Spanish Ministerio de Ciencia e Innovacion, and Agencia Estatal de Investigacion.

---
*Last updated: June 2025*

"""
This file contains all configurations and paths for the classifier models. Each classifier model is identified by its
domain (eg. defi). The TextClassifier.py script uses the configurations from this file to load all necessary files for
making text classifications.

Each model config contains three fields:
    - model_path : The path in which the trained XGBoost Classifier Model is stored.
    - vectorizer_path: The path in which the TFIDF Vectorizer is stored.
    - label_mappings: The class mappings for the classifications made by the model.
"""
import os

models_dir, vectorizers_dir = 'models/', 'vectorizers/'
model_configs = dict()

model_configs['defi'] = {
    'model_path': os.path.join(models_dir, 'xgb_defi.pkl'),
    'vectorizer_path': os.path.join(vectorizers_dir, 'tfidf_defi.pkl'),
    'label_mappings': {0: 'Cherry Swap', 1: 'Cryptocurrency', 2: 'Defi'}
}

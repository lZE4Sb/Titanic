import numpy as np
import pandas as pd
import mlflow
from omegaconf import OmegaConf
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.data import (
    create_splits,
    stateless_preprocessing,
    load_data,
    make_pipeline,
    prepare_data_by_folds
)
from src.env import seed_everything

from src.metrics import get_objective_metric, compute_metrics

import warnings
warnings.filterwarnings('ignore')

cfg = OmegaConf.load('config.yaml')

# DATA PREPARATION
df = load_data(cfg)
df = stateless_preprocessing(df)
pipeline = make_pipeline(cfg, df)
df = create_splits(cfg, df)

# Run training

model_dict = {
    'LogisticRegression': LogisticRegression(penalty=None),
    'L1_LogisticRegression': LogisticRegression(penalty='l1', solver='liblinear'),
    'L2_LogisticRegression': LogisticRegression(penalty='l2'),
    'ElasticNet_LogisticRegression': LogisticRegression(penalty='elasticnet', solver='saga'),
    'KNN': KNeighborsClassifier(n_neighbors=6),
    'DecisionTree': DecisionTreeClassifier(max_depth=6),
    'RandomForest': RandomForestClassifier(max_depth=10, n_estimators=200)}

# Run cycle
experiment_name = f"other_models_{datetime.now().strftime('%d%m%Y_%H%M')}"
mlflow.set_experiment(experiment_name)

for name, model in model_dict.items():
    with mlflow.start_run(run_name=f'{name}'):

        all_targets = []
        all_preds = [] # for oof rmsle

        #Folds cycle
        for fold in range(cfg.data.n_splits):
            seed_everything(cfg.general.seed + fold)
            # Data preparation
            X_train, y_train, X_val, y_val = prepare_data_by_folds(cfg, df, pipeline, fold)
            
            if cfg.data.log_transform_target:
                y_train = np.log1p(y_train)

            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            # Filling lists
            if cfg.data.log_transform_target:
                preds = np.expm1(preds)

            all_targets.extend(y_val)
            all_preds.extend(preds)

        obj_name, obj_spec = get_objective_metric(cfg)
        fold_score = compute_metrics(
            all_targets, all_preds, {obj_name: obj_spec}, y_proba=None)[obj_name] #OOF RMSLE
        
        mlflow.log_metric(f'{obj_name}', fold_score)
        print(f'{name}_{obj_name} - {fold_score}')

    
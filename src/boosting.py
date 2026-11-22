import os

import catboost as cat
import joblib
import lightgbm as lgbm
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import sklearn.metrics as skm
import xgboost as xgb
import yaml
from omegaconf import OmegaConf

from src.artifacts import (
    plot_confusion_matrix,
    plot_loss_curves,
    plot_optuna_study,
    plot_pr_curve,
    plot_predicted_vs_actual,
    plot_residuals,
    plot_roc_curve,
    plots_by_task
)
from src.data import prepare_data_by_folds
from src.env import seed_everything
from src.metrics import (
    compute_metrics,
    get_direction,
    get_metrics,
    get_objective_metric,
)
from src.predictions import predict_by_task

cfg = OmegaConf.load('config.yaml')

### MODEL DICT ###

MODEL_CLASSES = {
    ('xgb', 'regression'): xgb.XGBRegressor,
    ('xgb', 'binary'): xgb.XGBClassifier,
    ('xgb', 'multiclass'): xgb.XGBClassifier,
    
    ('lgbm', 'regression'): lgbm.LGBMRegressor,
    ('lgbm', 'binary'): lgbm.LGBMClassifier,
    ('lgbm', 'multiclass'): lgbm.LGBMClassifier,
    
    ('cat', 'regression'): cat.CatBoostRegressor,
    ('cat', 'binary'): cat.CatBoostClassifier,
    ('cat', 'multiclass'): cat.CatBoostClassifier,
}
### DATA PREPARATION ###

def prepare_cat_columns(df):
    cat_features = df.select_dtypes(include=[object]).columns.tolist()
    df[cat_features] = df[cat_features].fillna('missing').astype(str) 
    # idk, but looks like catboost cannot work with misses in categorical features
    df[cat_features] = df[cat_features].astype('category')

    return df, cat_features

### MODEL FUNCTIONS ###

def build_model_and_fit_params(cfg,
                               cat_features = None):
    '''
    Build model and fit parameters from config.
    Args:
        cfg: OmegaConf config with model_class, task_type, and model-specific settings
        cat_features: List of categorical feature indices/names (for CatBoost only)
    Returns:
        model: Instantiated model (based on task_type)
        fit_params: Dict of parameters for model.fit()
    Example:
        >>> model, fit_params = build_model_and_fit_params(cfg, cat_features=[0, 2, 5])
        >>> model.fit(X_train, y_train, , **fit_params)
    '''
    # building model
    model_class = cfg.model_class # lgbm | xgb | cat
    task_type = cfg.task_type     # regression | binary | multiclass

    key = (model_class, task_type)
    ModelClass = MODEL_CLASSES[key]
    model_cfg = cfg[model_class]  
    params = {**model_cfg.params}

    model = ModelClass(**params)

    #creating fit_params for specifiс libraries (lgbm and cat)
    fit_params = {**model_cfg.fit_params}

    if model_class == 'lgbm':
        callbacks = [
            lgbm.early_stopping(cfg.lgbm.callbacks.es_rounds),
            lgbm.log_evaluation(cfg.lgbm.callbacks.log_evaluation)]
        fit_params['callbacks'] = callbacks 
    
    if model_class == 'cat':
        fit_params['cat_features'] = cat_features
    
    return model, fit_params

### FOLD FUNCTIONS ###

def extract_losses(cfg, model):
    """Returns (train_losses, val_losses) regardless of library key naming."""
    model_class = cfg.model_class
    if model_class == 'xgb':
        evals = model.evals_result()
        train_key, val_key = 'validation_0', 'validation_1'
    elif model_class == 'lgbm':
        evals = model.evals_result_
        train_key = 'training' if 'training' in evals else next(iter(evals))
        val_key = next(k for k in evals if k != train_key)
    elif model_class == 'cat':
        evals = model.get_evals_result()
        train_key, val_key = 'learn', 'validation'
    else:
        raise ValueError(f'Unknown model: {model_class}')

    # Take metric name from evals
    val_metric = next(iter(evals[val_key]))
    train_metric = val_metric if val_metric in evals[train_key] else next(iter(evals[train_key]))
    
    return evals[train_key][train_metric], evals[val_key][val_metric]


def get_predictions_and_losses(cfg, X_train, y_train, X_val, y_val, model, fit_params):
    model_class = cfg.model_class
    fit_params = fit_params.copy()
    
    # Forming eval_set for different models
    if model_class in ('xgb', 'lgbm'):
        fit_params['eval_set'] = [(X_train, y_train), (X_val, y_val)]
    elif model_class == 'cat':
        fit_params['eval_set'] = (X_val, y_val)
    
    model.fit(X_train, y_train, **fit_params)

    # getting preds, proba, and losses
    y_pred, y_proba = predict_by_task(model, X_val, cfg)
    train_losses, val_losses = extract_losses(cfg, model)

    return y_pred, y_proba, train_losses, val_losses


def run_one_fold(cfg, df, pipeline, fold=0, cat_features=None):
    '''Run training on one fold
    Returns:
    model - for metrics computation in run_folds_cycle
    y_true, y_pred - in original units (if log transformation was used)
    y_proba - if possible
    fold_loss - metric for optimisation by Optuna (not used in optimisation just for computation of mean and std)
    '''
    seed_everything(cfg.general.seed + fold)
    
    X_train, y_train, X_val, y_val = prepare_data_by_folds(cfg, df, pipeline, fold)
    
    # log transformation for target
    if cfg.data.log_transform_target:
        y_train = np.log1p(y_train)
        y_val = np.log1p(y_val)

    ### Model building and getting fit parameters
    model, fit_params = build_model_and_fit_params(cfg, cat_features)
    
    #getting predictions, proba and losses
    y_pred, y_proba, train_losses, val_losses = get_predictions_and_losses(cfg, X_train, y_train, X_val, y_val, model, fit_params)
    
    # logging loss curves
    fig = plot_loss_curves(train_losses, val_losses, fold_idx=fold)
    plot_path = f'plots/fold{fold}_loss_curves.png'
    mlflow.log_figure(fig, plot_path)
    plt.close(fig)

    y_true = np.asarray(y_val).ravel()

    # Return to original target units
    if cfg.data.log_transform_target:
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)

    # getting optimization metric
    obj_name, obj_spec = get_objective_metric(cfg)
    fold_score = compute_metrics(
        y_true, y_pred, {obj_name: obj_spec}, y_proba=y_proba)[obj_name]

    return model, y_true, y_pred, y_proba, fold_score


def get_tree_number(cfg, model):
    """Number of boosting rounds to reuse for full training"""
    model_class = cfg.model_class
    if model_class == 'xgb':
        best = getattr(model, 'best_iteration', None)
        return best + 1 if best is not None else model.n_estimators
    elif model_class == 'lgbm':
        return model.best_iteration_ or model.n_estimators
    elif model_class == 'cat':
        best = model.get_best_iteration()
        return best + 1 if best is not None else model.tree_count_
    raise ValueError(f'Unknown model: {model_class}')


def run_folds_cycle(cfg, df, pipeline, trial=None, cat_features=None):
    all_targets, all_preds, all_probas = [], [], []
    all_losses, all_tree_numbers = [], []

    for fold in range(cfg.data.n_splits):
        model, y_true, y_pred, y_proba, fold_loss = run_one_fold(
            cfg, df, pipeline, fold, cat_features)

        if trial is not None:
            trial.report(fold_loss, fold)
            if trial.should_prune():
                print(f'Trial {trial.number} pruned at fold {fold}')
                raise optuna.TrialPruned()

        all_tree_numbers.append(get_tree_number(cfg, model))
        all_targets.append(y_true)
        all_preds.append(y_pred)
        all_probas.append(y_proba)
        all_losses.append(fold_loss)

    # OOF assembly
    oof_targets = np.concatenate(all_targets)
    oof_preds = np.concatenate(all_preds)
    if all_probas[0] is None:
        oof_proba = None
    elif all_probas[0].ndim == 1:      # binary: (n,) per fold
        oof_proba = np.concatenate(all_probas)
    else:                              # multiclass: (n, K) per fold
        oof_proba = np.vstack(all_probas)

    # metrics computation
    obj_name, _ = get_objective_metric(cfg)
    oof_metrics = compute_metrics(
        oof_targets, oof_preds, get_metrics(cfg), y_proba=oof_proba)
    oof_loss = oof_metrics[obj_name]

    mean_loss = float(np.mean(all_losses))
    loss_std = float(np.std(all_losses))

    # plots by task
    plots_by_task(cfg, oof_preds, oof_targets, oof_proba)

    results = {
        'oof_metrics': oof_metrics,
        'mean_loss': mean_loss,
        'loss_std': loss_std,
        'oof_loss': oof_loss,
        'all_tree_numbers': all_tree_numbers,
        'all_losses': all_losses,
        'oof_targets': oof_targets,
        'oof_preds': oof_preds,
        'oof_proba': oof_proba,
    }
    return results
    

### OPTUNA RUN ###

def suggest_trial_params(cfg, trial: optuna.Trial) -> dict:
    '''Creates a dict with trial parameters'''
    model_class = cfg.model_class
    task_type = cfg.task_type
    
    if model_class == 'xgb':
        trial_params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }
        if task_type == 'binary':
            trial_params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 1.0, 10.0)

    elif model_class == 'lgbm':
        trial_params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }
        if task_type == 'binary':
            trial_params['is_unbalance'] = trial.suggest_categorical('is_unbalance', [True, False])

    elif model_class == 'cat':
        trial_params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
        }
        if task_type == 'binary':
            trial_params['auto_class_weights'] = trial.suggest_categorical(
                'auto_class_weights', ['None', 'Balanced', 'SqrtBalanced'])
    else:
        raise ValueError(f'Unknown model: {model_class}')
    return trial_params


def objective(trial, base_cfg, df, pipeline, cat_features):

    # merging trial params and base config
    trial_params = suggest_trial_params(base_cfg, trial)
    model_class = base_cfg.model_class
    model_params = {model_class: {'params': trial_params}}
    cfg = OmegaConf.merge(base_cfg, OmegaConf.create(model_params))
    
    with mlflow.start_run(run_name=f'trial_{trial.number}'):
        
        mlflow.log_params(trial.params)

        #logging config
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        yaml_str = yaml.dump(cfg_dict, sort_keys=False, default_flow_style=False) #unless config dict saves in alphabetical order
        mlflow.log_text(yaml_str, f'trial_{trial.number}_config.yaml')

        # Folds cycle
        results = run_folds_cycle(cfg, df, pipeline, trial, cat_features)

        # logging metrics
        mean_loss = results['mean_loss']
        loss_std = results['loss_std']
        oof_loss = results['oof_loss']
        mlflow.log_metric('mean_val_loss', mean_loss)
        mlflow.log_metric('loss_std', loss_std)
        mlflow.log_metric('oof_loss', oof_loss)
        mlflow.log_metrics({f'oof_{k}': v for k, v in results['oof_metrics'].items()})

        # saving number of trees list
        all_tree_numbers = results['all_tree_numbers']
        trial.set_user_attr('all_tree_numbers', all_tree_numbers)
        mlflow.log_dict({'all_tree_numbers':all_tree_numbers}, 'all_tree_numbers.json')

    return oof_loss


def choose_num_iterations(cfg, best_num_iterations):
    strategy = cfg.result.num_iterations_strategy
    if strategy == 'mean':
        n = int(round(float(np.mean(best_num_iterations))))
    elif strategy == 'median':
        n = int(np.median(best_num_iterations))
    else:
        raise ValueError(f'Unknown strategy: {strategy}. Use mean or median.')
    return max(1, n) # to not return 0

def log_best_boosting_config(cfg, best_params, num_iterations=None):
    '''Creates full config from best params, number of epochs from optuna study 
    and base cfg
    best_params - study.best_params from optuna
    num_iterations - number of epochs/trees from choose_num_iterations 
    '''
    # Create nested dict from best_params
    nested = {}
    for key, value in best_params.items():
        parts = key.split('.')
        d = nested
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value

    model_class = cfg.model_class
    
    # adding tree number to model config
    if num_iterations is not None: 
        if model_class in ('lgbm', 'xgb'):
            nested['n_estimators'] = num_iterations
        elif model_class == 'cat':
            nested['iterations'] = num_iterations

    best_params = {model_class: {'params': nested}}
    best_cfg = OmegaConf.merge(cfg, OmegaConf.create(best_params))

    # mlflow logging
    cfg_dict = OmegaConf.to_container(best_cfg, resolve=True)
    yaml_str = yaml.dump(cfg_dict, sort_keys=False, default_flow_style=False) # sort_keys=False for sorting in original order
    mlflow.log_text(yaml_str, 'best_config.yaml')

    return best_cfg


def run_optuna_study(cfg, df, pipeline, study_name, cat_features=None, show_plots=False):
    best_config, num_trees = cfg, None # just for sure
    if cfg.optuna.enabled:
        study = optuna.create_study(
            study_name=study_name,
            direction=get_direction(cfg),
            sampler=optuna.samplers.TPESampler(seed=cfg.general.seed),
            pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=2,)
        )
        study.optimize(
            lambda trial: objective(trial, cfg, df, pipeline, cat_features),
            n_trials=cfg.optuna.n_trials,
            timeout=cfg.optuna.timeout,
            show_progress_bar=True,
        )
        # logging best study
        with mlflow.start_run(run_name='study_summary'):
            mlflow.log_params(study.best_params)
            mlflow.log_metric('best_loss', study.best_value)
            mlflow.log_metric('best_trial', study.best_trial.number)

            #logging number of trees for best trial
            trees_dict = {
                trial.number: trial.user_attrs['all_tree_numbers']
                for trial in study.trials
                if trial.state == optuna.trial.TrialState.COMPLETE}
            best_tree_numbers = trees_dict[study.best_trial.number]
            mlflow.log_dict({'best_tree_numbers':best_tree_numbers}, 'best_tree_numbers.json')

             # choosing number of trees for best config
            num_trees = choose_num_iterations(cfg, best_tree_numbers)

            # logging best config with choosed num of trees:
            best_config = log_best_boosting_config(cfg, study.best_params, num_trees)

            # logging study plots
            fig = plot_optuna_study(study, show=show_plots)
            mlflow.log_figure(fig, 'plots/optuna_history.png')
            plt.close(fig)

            # logging optuna study.pkl file
            if cfg.optuna.save_study: 
                study_path = f'{cfg.paths.output_dir}/studies'   
                os.makedirs(study_path, exist_ok=True)         
                joblib.dump(study, os.path.join(study_path, f'{study_name}.pkl'))
                
        print('\n' + '='*100)
        print('OPTUNA RESULTS')
        print('='*100)
        print('  Best trial:')
        print(f'  Value: {study.best_trial.value:.5f}')
        print(f'  Params: {study.best_trial.params}')
        print(f'  Number: {study.best_trial.number}')
        print(f'\nTotal trials: {len(study.trials)}')
    
    else:
        print('Optuna study disabled in config')

    return best_config, num_trees

### FULL TRAINING ###

# PREPARУ DATA FOR FULL TRAINING

def prepare_full_data(cfg, df, pipeline, test = None):
    '''Creates full training data, and test data if needed
    Returns X_train, y_train and X_test, y_test (optional)'''
    target_col = cfg.data.target_col

    # preparing full train data
    X_train = df.drop([target_col], axis=1)
    y_train = df[target_col]
    
    X_train = pipeline.fit_transform(X_train, y_train)

    if cfg.data.log_transform_target:
        y_train = np.log1p(y_train)

    if test is None:
        return X_train, y_train

    # preparing test data
    else:
        X_test = test.drop([target_col], axis=1)
        y_test = test[target_col]

        X_test = pipeline.transform(X_test)
        
        return X_train, y_train, X_test, y_test


def train_full_boosting(best_cfg, X_train, y_train, cat_features=None, num_trees = None):
    cfg = OmegaConf.create(OmegaConf.to_container(best_cfg, resolve=True))

    # deleting early stopping rounds parameter from config
    cfg['xgb']['params'].pop('early_stopping_rounds')
    cfg['lgbm']['callbacks'].pop('es_rounds')
    cfg['cat']['fit_params'].pop('early_stopping_rounds')

    # building model
    model_class = cfg.model_class # lgbm | xgb | cat
    task_type = cfg.task_type     # regression | binary | multiclass

    key = (model_class, task_type)
    ModelClass = MODEL_CLASSES[key]
    model_cfg = cfg[model_class]
    
    # adding number of trees
    if num_trees is not None:
        if model_class in ('lgbm', 'xgb'):
            model_cfg['params']['n_estimators'] = num_trees
        elif model_class == 'cat':
            model_cfg['params']['iterations'] = num_trees

    params = {**model_cfg.params}

    model = ModelClass(**params)

    #creating fit_params for specifiс libraries (lgbm and cat)
    fit_params = {**model_cfg.fit_params}

    if model_class == 'lgbm':
        callbacks = [
            lgbm.log_evaluation(cfg.lgbm.callbacks.log_evaluation),]
        fit_params['callbacks'] = callbacks 
    
    if model_class == 'cat':
        fit_params['cat_features'] = cat_features

    # training model
    model.fit(X_train, y_train, **fit_params)

    return model


def final_evaluation(cfg, model, X_test, y_test):
    '''Evaluate model on test set using all task metrics.'''
    y_pred, y_proba = predict_by_task(model, X_test, cfg)
    
    metrics = compute_metrics(
        y_test, y_pred, get_metrics(cfg), y_proba=y_proba)
    
    obj_name, _ = get_objective_metric(cfg)
    
    # Log all metrics
    mlflow.log_metrics({f'test_{k}': v for k, v in metrics.items()})
    print(f'Test {obj_name}: {metrics[obj_name]:.4f}')
    return metrics


def save_boosting_model(cfg, model):
    """Save boosting model to disk and log to MLflow."""
    model_extensions = {
    'xgb': '.json',
    'lgbm': '.txt',
    'cat': '.cbm',
}
    model_class = cfg.model_class
    ext = model_extensions[model_class]
    model_path = f"model_{model_class}{ext}"

    if model_class == 'xgb':
        model.save_model(model_path)
    elif model_class == 'lgbm':
        model.booster_.save_model(model_path)
    elif model_class == 'cat':
        model.save_model(model_path, format='cbm')

    mlflow.log_artifact(model_path, artifact_path="model")
    os.remove(model_path)
    print(f"Model saved and logged: {model_path}")
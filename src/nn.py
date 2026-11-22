import os

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import torch
import yaml
from omegaconf import OmegaConf
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from src.artifacts import (
    plot_confusion_matrix,
    plot_loss_curves,
    plot_optuna_study,
    plot_pr_curve,
    plot_predicted_vs_actual,
    plot_residuals,
    plot_roc_curve,
    plots_by_task,
)
from src.data import prepare_data_by_folds
from src.env import seed_everything
from src.metrics import (
    compute_metrics,
    get_direction,
    get_metrics,
    get_objective_metric,
)
from src.predictions import predict_by_task_torch

cfg = OmegaConf.load('config.yaml')

##### MODEL CREATION FUNCTIONS #####

def create_dataloaders(cfg, X_train, y_train, X_val=None, y_val=None):
    '''Creates DataLoader(s). Returns (train_loader,) or (train_loader, val_loader).'''
    
    # Convert DataFrame to numpy
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if X_val is not None and hasattr(X_val, 'values'):
        X_val = X_val.values
    
    # Convert target to 1d array
    if hasattr(y_train, 'values'):
        y_train = y_train.values.ravel()
    if y_val is not None and hasattr(y_val, 'values'):
        y_val = y_val.values.ravel()
    
    # Convert to tensors
    task_type = cfg.task_type
    X_train_t = torch.FloatTensor(X_train)
    if task_type == 'multiclass':
        y_train_t = torch.LongTensor(y_train)
    else:
        y_train_t = torch.FloatTensor(y_train)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    
    # Shuffle generator for reproducibility
    g = torch.Generator()
    g.manual_seed(cfg.general.seed)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.nn.data.batch_size,
        shuffle=True,
        num_workers=cfg.nn.data.num_workers,
        generator=g)
    
    # Return only train loader if no val/test provided
    if X_val is None or y_val is None:
        return train_loader
    
    # Create val/test loader (shuffle always False for evaluation)
    X_val_t = torch.FloatTensor(X_val)
    if task_type == 'multiclass':
        y_val_t = torch.LongTensor(y_val)
    else:
        y_val_t = torch.FloatTensor(y_val)

    val_dataset = TensorDataset(X_val_t, y_val_t)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.nn.data.batch_size,
        shuffle=False,
        num_workers=cfg.nn.data.num_workers)
    
    return train_loader, val_loader


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, activation, output_dim=1):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            in_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)  # shape: (batch_size, output_dim)


def create_optimizer(model, cfg):
    '''Creates optimizer from config'''
    opt_name = cfg.nn.training.optimizer
    
    if opt_name == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=cfg.nn.training.lr,
            weight_decay=cfg.nn.training.weight_decay
        )
    elif opt_name == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=cfg.nn.training.lr,
            weight_decay=cfg.nn.training.weight_decay,
            momentum=cfg.nn.training.momentum
        )
    elif opt_name == 'rmsprop':
        return optim.RMSprop(
            model.parameters(),
            lr=cfg.nn.training.lr,
            weight_decay=cfg.nn.training.weight_decay,
            momentum=cfg.nn.training.momentum
        )
    else:
        raise ValueError(f'Unknown optimizer: {opt_name}')

def create_scheduler(optimizer, cfg, num_epochs=None):
    sched_name = cfg.nn.training.scheduler
    t_max = num_epochs if num_epochs is not None else cfg.nn.training.epochs

    if sched_name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=cfg.nn.training.min_lr,
        )
    elif sched_name == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=cfg.nn.training.scheduler_factor,
            patience=cfg.nn.training.scheduler_patience,
            min_lr=cfg.nn.training.min_lr,
        )
    elif sched_name == 'none':
        return None
    else:
        raise ValueError(f'Unknown scheduler: {sched_name}')


def create_loss_fn(cfg):
    '''Creates loss from config'''
    loss_name = cfg.nn.training.loss if hasattr(cfg.nn.training, 'loss') else 'mse'
    
    if loss_name == 'mse':
        return nn.MSELoss()
    elif loss_name == 'mae':
        return nn.L1Loss()
    elif loss_name == 'rmse':
        return lambda pred, target: torch.sqrt(nn.MSELoss()(pred, target) + 1e-8)
    elif loss_name == 'crossentropy':
        return nn.CrossEntropyLoss()
    elif loss_name == 'bce':
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f'Unknown loss: {loss_name}')

##### MODEL TRAINING FUNCTIONS #####

def train_nn(model, train_loader, val_loader, optimizer, loss_fn, cfg):
    '''Training loop with early stopping. Returns model, best_val_loss, loss histories and epoch, where training stopped'''
    device = torch.device(cfg.nn.training.device)
    model = model.to(device)

    scheduler = create_scheduler(optimizer, cfg)
    epochs = cfg.nn.training.epochs
    patience = cfg.nn.training.patience
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    # Track loss histories
    train_losses = []
    val_losses = []

    final_epoch = 0

    for epoch in range(epochs):
        #Epoch counter
        final_epoch = epoch + 1

        # Training
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            y_pred = y_pred.squeeze(-1)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = loss_fn(y_pred.squeeze(-1), y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # LR Scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)   # reacts on loss
            else:
                scheduler.step() # reacts by itself

        # MLflow logging
        mlflow.log_metric('train_loss', train_loss, step=epoch)
        mlflow.log_metric('val_loss', val_loss, step=epoch)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # print(f'Early stopping at epoch {epoch}')
                break

        # Verbose params    
        verbose = cfg.nn.training.verbose
        if (epoch + 1) % verbose == 0:
            print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # Loading best model
    if best_state is not None:
        model.load_state_dict(best_state) 
           
    return model, best_val_loss, train_losses, val_losses, final_epoch


def run_one_fold(cfg, df, pipeline, fold=0):
    '''Train one fold. Returns model, val_loss, predictions, targets, loss histories and final epoch'''
    seed_everything(cfg.general.seed + fold)
    
    X_train, y_train, X_val, y_val = prepare_data_by_folds(cfg, 
        df, pipeline, fold)

    if cfg.data.log_transform_target:
        y_train = np.log1p(y_train)
        y_val = np.log1p(y_val)
    
    train_loader, val_loader = create_dataloaders(
        cfg, X_train, y_train, X_val, y_val)

    model = MLP(**cfg.nn.model)
    
    optimizer = create_optimizer(model, cfg)
    loss_fn = create_loss_fn(cfg)

    model, best_fold_val_loss, train_losses, val_losses, final_epoch = train_nn(
        model, train_loader, val_loader, optimizer, loss_fn, cfg)
    
    # Get predictions and logits on validation set for plots
    device = torch.device(cfg.nn.training.device)
    model = model.to(device)
    model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch).squeeze(-1)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(y_batch.numpy())
    logits = np.concatenate(all_logits)
    targets = np.concatenate(all_targets)

    # Convert logits to predictions by task
    y_pred, y_proba = predict_by_task_torch(logits, cfg)

    # logging loss curves
    fig = plot_loss_curves(train_losses, val_losses, fold_idx=fold)
    plot_path = f'plots/fold{fold}_loss_curves.png'
    mlflow.log_figure(fig, plot_path)
    plt.close(fig)

    return model, best_fold_val_loss, y_pred, y_proba, targets, train_losses, val_losses, final_epoch


def run_folds_cycle(cfg, df, pipeline, trial=None):
    fold_losses = []
    all_y_pred, all_y_proba, all_targets = [], [], []
    all_train_losses, all_val_losses = [], []
    num_epochs = []

    for fold in range(cfg.data.n_splits):
        model, best_fold_val_loss, y_pred, y_proba, targets, train_losses, val_losses, final_epoch = run_one_fold(
            cfg, df, pipeline, fold)
        fold_losses.append(best_fold_val_loss)
        all_y_pred.append(y_pred)
        all_y_proba.append(y_proba)
        all_targets.append(targets)
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        num_epochs.append(final_epoch)

        # prunung
        if trial is not None:
            score = best_fold_val_loss
            if get_direction(cfg) == 'maximize':
                score = -score   # reversing loss for pruner
            trial.report(score, fold)
            if trial.should_prune():
                print(f'Trial {trial.number} pruned at fold {fold}')
                raise optuna.TrialPruned()

    # OOF assembly - binary proba is 1D, multiclass is 2D
    oof_y_pred = np.concatenate(all_y_pred)
    oof_targets = np.concatenate(all_targets)
    if all_y_proba[0] is None:
        oof_proba = None
    elif all_y_proba[0].ndim == 1:
        oof_proba = np.concatenate(all_y_proba)
    else:
        oof_proba = np.vstack(all_y_proba)

    mean_loss = float(np.mean(fold_losses))
    loss_std = float(np.std(fold_losses))

    plots_by_task(cfg, oof_y_pred, oof_targets, oof_proba)

    results = {
        'mean_loss': mean_loss,
        'loss_std': loss_std,
        'fold_losses': fold_losses,
        'oof_y_pred': oof_y_pred,
        'oof_proba': oof_proba,
        'oof_targets': oof_targets,
        'train_losses': all_train_losses,
        'val_losses': all_val_losses,
        'num_epochs': num_epochs
    }
    return results

##### OPTUNA FUNCTIONS #####

def suggest_trial_params(trial: optuna.Trial) -> dict:
    '''Creates a dict with trial parameters'''
    optimizer_name = trial.suggest_categorical(
        'nn.training.optimizer', ['adam', 'sgd', 'rmsprop']
    )
    activation = trial.suggest_categorical(
        'nn.model.activation', ['relu', 'tanh']
    )
    trial_params = {
        'nn': {
            'model': {
                'hidden_dim': trial.suggest_categorical(
                    'nn.model.hidden_dim', [32, 64, 128, 256]),
                'num_layers': trial.suggest_int('nn.model.num_layers', 1, 5),
                'dropout': trial.suggest_float('nn.model.dropout', 0.0, 0.5),
                'activation': activation,
                    },
            'training': {
                'optimizer': optimizer_name,
                'lr': trial.suggest_float(
                    'nn.training.lr', 1e-5, 1e-2, log=True),
                'weight_decay': trial.suggest_float(
                    'nn.training.weight_decay', 1e-6, 1e-2, log=True),
                        },
            'data': {
                'batch_size': trial.suggest_categorical(
                    'nn.data.batch_size', [32, 64, 128, 256]),
                    },
            }
    }
    if optimizer_name in ('sgd', 'rmsprop'):
        trial_params['nn']['training']['momentum'] = trial.suggest_float(
            'nn.training.momentum', 0.8, 0.99
        )
    return trial_params


def choose_num_iterations(cfg, best_num_iterations):
    strategy = cfg.result.num_iterations_strategy
    if strategy == 'mean':
        n = int(round(float(np.mean(best_num_iterations))))
    elif strategy == 'median':
        n = int(np.median(best_num_iterations))
    else:
        raise ValueError(f'Unknown strategy: {strategy}. Use mean or median.')
    return max(1, n)


def log_best_config(cfg, best_params, num_iterations=None):
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

        best_cfg = OmegaConf.merge(cfg, OmegaConf.create(nested))
        if num_iterations is not None:
            best_cfg.nn.training.epochs = num_iterations

        # mlflow logging
        cfg_dict = OmegaConf.to_container(best_cfg, resolve=True)
        yaml_str = yaml.dump(cfg_dict, sort_keys=False, default_flow_style=False) # sort_keys=False for sorting in original order
        mlflow.log_text(yaml_str, 'best_config.yaml')

    return best_cfg


def objective(trial: optuna.Trial, base_cfg, df, pipeline) -> float:
    
    # Create trial config
    trial_params = suggest_trial_params(trial)
    cfg = OmegaConf.merge(base_cfg, OmegaConf.create(trial_params))

    with mlflow.start_run(run_name=f'trial_{trial.number}'):
        mlflow.log_params(trial.params)

        #logging config
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        yaml_str = yaml.dump(cfg_dict, sort_keys=False, default_flow_style=False) #unless config dict saves in alphabetical order
        mlflow.log_text(yaml_str, f'trial_{trial.number}_config.yaml')

        # Folds cycle
        results = run_folds_cycle(cfg, df, pipeline, trial)

        # logging metrics
        oof_preds = results['oof_y_pred']
        oof_targets = results['oof_targets']
        oof_proba = results['oof_proba']

        if cfg.data.log_transform_target:
            oof_preds = np.expm1(oof_preds)
            oof_targets = np.expm1(oof_targets)

        oof_metrics = compute_metrics(oof_targets, oof_preds, get_metrics(cfg), y_proba=oof_proba)
        obj_name, _ = get_objective_metric(cfg)
        oof_loss = oof_metrics[obj_name]
        mlflow.log_metrics({f'oof_{k}': v for k, v in oof_metrics.items()})

        num_epochs = results['num_epochs']
        trial.set_user_attr('num_epochs', num_epochs)
    return oof_loss


def run_optuna_study(cfg, df, pipeline, study_name, show_plots=False):
    best_config, best_num_epochs = cfg, None
    if cfg.optuna.enabled:
        study = optuna.create_study(
            study_name=study_name,
            direction=get_direction(cfg),
            sampler=optuna.samplers.TPESampler(seed=cfg.general.seed),
            pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=2,),
            storage=None,
            load_if_exists=True,
        )
        study.optimize(
            lambda trial: objective(trial, cfg, df, pipeline),
            n_trials=cfg.optuna.n_trials,
            timeout=cfg.optuna.timeout,
            show_progress_bar=True
        )
        # logging data about best trial
        with mlflow.start_run(run_name='study_summary'):
            mlflow.log_params(study.best_params)
            mlflow.log_metric('best_loss', study.best_value)
            mlflow.log_metric('best_trial', study.best_trial.number)

            #logging number of epochs for best trial
            epochs_dict = {
                trial.number: trial.user_attrs['num_epochs']
                for trial in study.trials
                if trial.state == optuna.trial.TrialState.COMPLETE}
            best_num_epochs = epochs_dict[study.best_trial.number]
            mlflow.log_dict({'best_num_epochs':best_num_epochs}, 'best_num_epochs.json')

            # choosing number of epochs for best config
            num_epochs = choose_num_iterations(cfg, best_num_epochs)

            # logging best config with choosed num of epochs:
            best_config = log_best_config(cfg, study.best_params, num_epochs)

            # logging study plots
            fig = plot_optuna_study(study, show=show_plots)
            mlflow.log_figure(fig, 'plots/optuna_history.png')
            plt.close(fig)

            # logging optuna study.pkl file
            if cfg.optuna.save_study: 
                study_path = f'{cfg.paths.output_dir}/studies'   
                os.makedirs(study_path, exist_ok=True)         
                joblib.dump(study, os.path.join(study_path, f'{study_name}.pkl'))

        print(f'Best trial: {study.best_trial.number}')
        print(f'Best value: {study.best_value:.4f}')
        print(f'Best params: {study.best_params}')

    else:
        print('Optuna study disabled in config')

    return best_config, best_num_epochs # best params for full training

##### FULL TRAINING #####

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

        if cfg.data.log_transform_target:
            y_test = np.log1p(y_test)
        
        return X_train, y_train, X_test, y_test
    

def train_final_nn(model, train_loader, optimizer, loss_fn, cfg, num_epochs=None):
    '''Training loop without validation or early stopping. Returns model and train_losses.
    num_epochs - if missing, taken from config'''
    device = torch.device(cfg.nn.training.device)
    model = model.to(device)
    verbose = cfg.nn.training.verbose

    if num_epochs is None:
        num_epochs = cfg.nn.training.epochs

    # Scheduler for final training - only cosine, because no validation available
    scheduler = create_scheduler(optimizer, cfg, num_epochs=num_epochs)
    
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze(-1)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Scheduler step - cosine/step work per epoch, plateau needs a metric
        if scheduler is not None and not isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
        # MLflow logging
        mlflow.log_metric('train_loss', train_loss, step=epoch)
        mlflow.log_metric('num_epochs', num_epochs)
        
        if (epoch + 1) % verbose == 0:
            print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}')
    
    return model, train_losses


def final_evaluation(cfg, model, test_loader):
    '''Evaluate model on test dataloader'''
    device = torch.device(cfg.nn.training.device)
    model = model.to(device)
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch).squeeze(-1)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(y_batch.numpy())
    logits = np.concatenate(all_logits)
    targets = np.concatenate(all_targets)

    y_pred, y_proba = predict_by_task_torch(logits, cfg)

    if cfg.data.log_transform_target and cfg.task_type == 'regression':
        y_pred = np.expm1(y_pred)
        targets = np.expm1(targets)

    metrics = compute_metrics(targets, y_pred, get_metrics(cfg), y_proba=y_proba)
    obj_name, _ = get_objective_metric(cfg)

    mlflow.log_metrics({f'test_{k}': v for k, v in metrics.items()})
    print(f'Test {obj_name}: {metrics[obj_name]:.4f}')
    return metrics


def save_model(model):
    model_path = 'model.pth'
    torch.save(model.state_dict(), model_path)
    mlflow.log_artifact(model_path)
    os.remove(model_path)  # cleanup
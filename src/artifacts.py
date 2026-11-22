import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import numpy as np
import optuna
import optuna.importance
from sklearn import metrics as skm
import mlflow

def plots_by_task(cfg, oof_preds, oof_targets, oof_proba = None):
    task_type = cfg.task_type
    
    if task_type == 'regression':
        fig1 = plot_predicted_vs_actual(oof_preds, oof_targets)
        mlflow.log_figure(fig1, 'plots/predicted_vs_actual.png')
        plt.close(fig1)
        fig2 = plot_residuals(oof_preds, oof_targets)
        mlflow.log_figure(fig2, 'plots/residuals.png')
        plt.close(fig2)
    else: # for multiclass and binary
        fig_cm = plot_confusion_matrix(oof_targets, oof_preds)
        mlflow.log_figure(fig_cm, 'plots/confusion_matrix.png')
        plt.close(fig_cm)
        # only for binary
        if cfg.task_type == 'binary':
            fig_roc = plot_roc_curve(oof_targets, oof_proba)
            mlflow.log_figure(fig_roc, 'plots/roc_curve.png')
            plt.close(fig_roc)
            fig_pr = plot_pr_curve(oof_targets, oof_proba)
            mlflow.log_figure(fig_pr, 'plots/pr_curve.png')
            plt.close(fig_pr)


def plot_loss_curves(train_losses_list, val_losses_list, fold_idx=None):
    '''Plot training and validation loss curves for one or all folds.'''
    fig = plt.figure(figsize=(10, 6))
    
    if fold_idx is not None:
        # Single fold
        epochs = range(len(train_losses_list))
        plt.plot(epochs, train_losses_list, label='Train Loss', linewidth=2)
        plt.plot(epochs, val_losses_list, label='Validation Loss', linewidth=2)
        plt.title(f'Loss Curves - Fold {fold_idx}')
    else:
        # All folds
        for i, (train_loss, val_loss) in enumerate(zip(train_losses_list, val_losses_list)):
            epochs = range(len(train_loss))
            plt.plot(epochs, train_loss, alpha=0.5, label=f'Train Fold {i}')
            plt.plot(epochs, val_loss, alpha=0.5, label=f'Val Fold {i}')
        plt.title('Loss Curves - All Folds')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig

def plot_predicted_vs_actual(predictions_list, targets_list):
    '''Scatter plot of predictions vs actual values across all folds.'''
    all_preds = np.asarray(predictions_list).ravel()
    all_targets = np.asarray(targets_list).ravel()
    
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(all_targets, all_preds, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(all_targets.min(), all_preds.min())
    max_val = max(all_targets.max(), all_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig

def plot_residuals(predictions_list, targets_list):
    '''Plot residual distribution and residuals vs predictions.'''
    all_preds = np.asarray(predictions_list).ravel()
    all_targets = np.asarray(targets_list).ravel()
    residuals = all_targets - all_preds
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residual distribution
    ax1.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero line')
    ax1.set_xlabel('Residuals (Actual - Predicted)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Residual Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residuals vs predictions
    ax2.scatter(all_preds, residuals, alpha=0.5, s=20)
    ax2.axhline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals vs Predicted')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def plot_confusion_matrix(y_true, y_pred):
    cm = skm.confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_proba):
    fpr, tpr, _ = skm.roc_curve(y_true, y_proba)
    auc = skm.roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'r--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_pr_curve(y_true, y_proba):
    precision, recall, _ = skm.precision_recall_curve(y_true, y_proba)
    ap = skm.average_precision_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, label=f'PR (AP = {ap:.3f})', linewidth=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_optuna_study(study, show=True):
    '''Plot Optuna optimization history and parameter importance.
    Returns matplotlib Figure for MLflow logging.
    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Optimization history
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trial_numbers = [t.number for t in trials]
    trial_values = [t.value for t in trials]

    trial_values = np.array(trial_values)
    if study.direction == optuna.study.StudyDirection.MAXIMIZE:
        best_values = np.maximum.accumulate(trial_values)
    else:
        best_values = np.minimum.accumulate(trial_values)

    ax1.scatter(trial_numbers, trial_values, alpha=0.6, s=30)
    ax1.plot(trial_numbers, trial_values, alpha=0.3, linewidth=1)

    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('Objective Value')
    ax1.set_title('Optimization History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Parameter importance with Fanova 
    if len(trials) >= 10:
        try:
            importance = optuna.importance.get_param_importances(study)
            sorted_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            param_names = [p[0] for p in sorted_params[:10]]
            param_scores = [p[1] for p in sorted_params[:10]]

            ax2.barh(param_names, param_scores, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Importance Score')
            ax2.set_title('Parameter Importance (Fanova)')
            ax2.grid(True, alpha=0.3, axis='x')
        except Exception as e:
            ax2.text(0.5, 0.5, f'Importance unavailable:\n{e}',
                     ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Parameter Importance (N/A)')
    else:
        ax2.text(0.5, 0.5, 'Need >= 10 completed trials',
                 ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Parameter Importance (N/A)')

    plt.tight_layout()

    if show:
        plt.show()

    return fig

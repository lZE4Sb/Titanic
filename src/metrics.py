import numpy as np
from sklearn import metrics as skm


def _rmse(y, p):
    return float(np.sqrt(skm.mean_squared_error(y, p)))


def _rmsle(y, p):
    p = np.clip(np.asarray(p, dtype=float), 0, None)  # RMSLE undefined for negative preds
    return float(np.sqrt(skm.mean_squared_log_error(y, p)))


def _wape(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    denom = np.abs(y).sum()
    return float(np.abs(y - p).sum() / denom) if denom > 0 else float('nan')


# fn       - metric function
# proba    - True: takes probabilities, False: takes hard predictions
# maximize - True: higher is better (drives Optuna direction)
# kwargs   - extra arguments for fn
METRICS_BY_TASK = {
    'regression': {
        'rmse':  {'fn': _rmse, 'proba': False, 'maximize': False},
        'rmsle': {'fn': _rmsle, 'proba': False, 'maximize': False},
        'mae':   {'fn': skm.mean_absolute_error, 'proba': False, 'maximize': False},
        'r2':    {'fn': skm.r2_score, 'proba': False, 'maximize': True},
        'wape':  {'fn': _wape, 'proba': False, 'maximize': False},
    },
    'binary': {
        'roc_auc':  {'fn': skm.roc_auc_score, 'proba': True, 'maximize': True},
        'pr_auc':   {'fn': skm.average_precision_score, 'proba': True, 'maximize': True},
        'log_loss': {'fn': skm.log_loss, 'proba': True, 'maximize': False},
        'f1':       {'fn': skm.f1_score, 'proba': False, 'maximize': True,
                     'kwargs': {'zero_division': 0}},
        'accuracy': {'fn': skm.accuracy_score, 'proba': False, 'maximize': True},
    },
    'multiclass': {
        'log_loss': {'fn': skm.log_loss, 'proba': True, 'maximize': False},
        'roc_auc':  {'fn': skm.roc_auc_score, 'proba': True, 'maximize': True,
                     'kwargs': {'multi_class': 'ovr', 'average': 'macro'}},
        'f1_macro': {'fn': skm.f1_score, 'proba': False, 'maximize': True,
                     'kwargs': {'average': 'macro', 'zero_division': 0}},
        'f1_micro': {'fn': skm.f1_score, 'proba': False, 'maximize': True,
                             'kwargs': {'average': 'micro', 'zero_division': 0}},
        'accuracy': {'fn': skm.accuracy_score, 'proba': False, 'maximize': True},
        'balanced_accuracy': {'fn': skm.balanced_accuracy_score, 'proba': False, 'maximize': True},
    },
}

_LABELS_AWARE = {'log_loss', 'f1', 'f1_macro'}  # metrics accepting `labels`


def get_metrics(cfg):
    """All metrics available for the current task."""
    return METRICS_BY_TASK[cfg.task_type]


def get_objective_metric(cfg):
    """Returns (name, spec) of the metric Optuna optimizes."""
    available = get_metrics(cfg)
    name = cfg.optuna.metric
    if name not in available:
        raise ValueError(
            f"Metric '{name}' is not available for task '{cfg.task_type}'. "
            f"Available: {list(available)}")
    return name, available[name]


def get_direction(cfg):
    """Optuna direction derived from the objective metric."""
    _, spec = get_objective_metric(cfg)
    return 'maximize' if spec['maximize'] else 'minimize'


def compute_metrics(y_true, y_pred, metrics, y_proba=None, labels=None):
    """Compute metrics from predictions regarding on model_type (boosting and torch).

    y_pred  - hard labels (classification) or values (regression)
    y_proba - (n,) positive class proba for binary, (n, K) for multiclass, None for regression
    labels  - full class list, protects metrics when a fold misses a class
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    results = {}
    for name, spec in metrics.items():
        kwargs = dict(spec.get('kwargs', {}))
        if labels is not None and name in _LABELS_AWARE and 'labels' not in kwargs:
            kwargs['labels'] = labels

        target = y_proba if spec['proba'] else y_pred
        if spec['proba'] and target is None:
            results[name] = float('nan')
            continue
        try:
            results[name] = float(spec['fn'](y_true, target, **kwargs))
        except ValueError as e:
            print(f'[metric {name}] skipped: {e}')
            results[name] = float('nan')
    return results
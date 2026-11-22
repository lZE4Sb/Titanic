import numpy as np


def predict_by_task(model, X, cfg, threshold=0.5):
    """Unified inference. Returns (y_pred, y_proba).
    regression -> (values, None)
    binary     -> (labels, proba of class 1),  proba shape (n,)
    multiclass -> (labels, proba matrix),      proba shape (n, K)
    """
    task_type = cfg.task_type

    if task_type == 'regression':
        return np.asarray(model.predict(X)).ravel(), None

    proba = model.predict_proba(X)

    if task_type == 'binary':
        proba = proba[:, 1]
        y_pred = (proba >= threshold).astype(int)
    elif task_type == 'multiclass':
        y_pred = proba.argmax(axis=1)
    else:
        raise ValueError(f'Unknown task_type: {task_type}')

    return np.asarray(y_pred).ravel(), proba

def predict_by_task_torch(logits, cfg, threshold=0.5):
    """Unified inference for torch models from raw logits.
    Returns (y_pred, y_proba).
    regression -> (values, None)
    binary     -> (labels, proba of class 1),  proba shape (n,)
    multiclass -> (labels, proba matrix),      proba shape (n, K)
    """
    task_type = cfg.task_type
    logits = np.asarray(logits)

    if task_type == 'regression':
        return logits.ravel(), None

    if task_type == 'binary':
        proba = 1.0 / (1.0 + np.exp(-logits.ravel()))  # sigmoid
        y_pred = (proba >= threshold).astype(int)
    elif task_type == 'multiclass':
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        proba = exp / exp.sum(axis=1, keepdims=True)  # softmax
        y_pred = proba.argmax(axis=1)
    else:
        raise ValueError(f'Unknown task_type: {task_type}')

    return y_pred.ravel(), proba
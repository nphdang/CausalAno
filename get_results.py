import numpy as np
from sklearn import metrics

def tabular_metrics(y_true, y_score):
    """
    Calculates evaluation metrics for tabular anomaly detection.
    Adapted from  https://github.com/xuhongzuo/DeepOD/blob/main/deepod/metrics/_anomaly_detection.py
    Args:

        y_true (np.array, required):
            Data label, 0 indicates normal timestamp, and 1 is anomaly.

        y_score (np.array, required):
            Predicted anomaly scores, higher score indicates higher likelihoods to be anomaly.

    Returns:
        tuple: A tuple containing:

        - auc_roc (float):
            The score of area under the ROC curve.

        - auc_pr (float):
            The score of area under the precision-recall curve.

        - f1 (float):
            The score of F1-score.

        - precision (float):
            The score of precision.

        - recall (float):
            The score of recall.

    """
    # F1@k, using real percentage to calculate F1-score
    n_test = len(y_true)
    new_index = np.random.permutation(
        n_test)  # shuffle y to prevent bias of ordering (argpartition may discard entries with same value)
    y_true = y_true[new_index]
    y_score = y_score[new_index]

    top_k = len(np.where(y_true == 1)[0])
    indices = np.argpartition(y_score, -top_k)[-top_k:]
    y_pred = np.zeros_like(y_true)
    y_pred[indices] = 1

    y_true = y_true.astype(int)
    p, r, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred, average='binary')

    return metrics.accuracy_score(y_true, y_pred), f1, metrics.roc_auc_score(y_true, y_score),


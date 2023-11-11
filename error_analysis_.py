import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, roc_curve, f1_score
from sklearn.metrics import recall_score, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(y, y_pred):
    """Confusion matrix plotting in matplotlib style"""
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(colorbar=False)
    plt.show()


def plot_precision_recall(y, y_pred, versus=False):
    """Plotting of precision and recall curve
    versus: If True, plots Precision x Recall curve
            If False, plots Precision and Recall x thresholds curve"""
    precisions, recalls, thresholds = precision_recall_curve(y, y_pred)
    if not versus:
        plt.title('Precision and Recall x Thresholds')
        plt.xlabel('Thresholds')
        plt.ylabel('Precision and Recall')
        plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
        plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
        plt.legend()
    else:
        plt.title('Precision x Recall')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.plot(recalls[:-1], precisions[:-1], 'k--')
    plt.show()


def plot_roc_curve(y, y_pred):
    """Receiver Operating Curve plotting"""
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    plt.plot(fpr, tpr, linewidth=2)
    plt.title('Receiver Operating Curve')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.show()


def new_treshold_metrics(y, y_pred, metric='precision', value=1):
    """Calculation of all classification metrics with precision or recall fixed by user
    metric: 'precision' or 'recall', the intended parameter
    value: value of the intended parameter"""
    precisions, recalls, thresholds = precision_recall_curve(y, y_pred)
    if metric == 'precision':
        threshold = thresholds[np.argmax(precisions >= value)]
    elif metric == 'recall':
        print(np.argmax(recalls <= value))
        threshold = thresholds[np.argmax(recalls <= value)]
        print(threshold)
    else:
        raise ValueError(f'{metric} não é uma métrica válida! Métricas aceitas: precision e recall')
    y_pred_new = (y_pred >= threshold)
    print(f'Classification report:\n'
          f'Accuracy: {round(accuracy_score(y, y_pred_new), 5)}\n'
          f'Precision: {round(precision_score(y, y_pred_new), 5)}\n'
          f'Recall: {round(recall_score(y, y_pred_new), 5)}\n'
          f'F1: {round(f1_score(y, y_pred_new), 5)}\n'
          f'AUC: {round(roc_auc_score(y, y_pred_new), 5)}\n')

    return y_pred_new


def metrics(y, y_pred):
    """Calculation of classification metrics"""
    print(f'Classification report:\n'
          f'Accuracy: {round(accuracy_score(y, y_pred), 5)}\n'
          f'Precision: {round(precision_score(y, y_pred), 5)}\n'
          f'Recall: {round(recall_score(y, y_pred), 5)}\n'
          f'F1: {round(f1_score(y, y_pred), 5)}\n'
          f'AUC: {round(roc_auc_score(y, y_pred), 5)}\n')
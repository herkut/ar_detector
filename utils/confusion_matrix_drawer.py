import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def concatenate_classification_reports(report1, report2):
    results = {}

    TP = report1['TP'] + report2['TP']
    FP = report1['FP'] + report2['FP']
    TN = report1['TN'] + report2['TN']
    FN = report1['FN'] + report2['FN']

    results['TP'] = TP
    results['FP'] = FP
    results['TN'] = TN
    results['FN'] = FN
    results['sensitivity/recall'] = TP / ((TP + FN) if (TP + FN) > 0 else 1)
    results['specificity'] = TN / ((TN + FP) if (TN + FP) > 0 else 1)
    results['precision'] = TP / ((TP + FP) if (TP + FP) > 0 else 1)
    results['accuracy'] = (TP + TN) / (TP + FN + TN + FP)
    results['f1'] = 2 * TP / (2 * TP + FP + FN)

    return results


def get_class_from_probability(y_pred):
    return 1 if torch.sigmoid(y_pred) > 0.5 else 0


def classification_report(y_true, y_pred):
    results = {}

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    if isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor):
        for i in range(len(y_pred)):
            if y_true[i] == y_pred[i] == 1:
                TP += 1
            if y_pred[i] == 1 and y_true[i] != y_pred[i]:
                FP += 1
            if y_true[i] == y_pred[0] == 0:
                TN += 1
            if y_pred[i] == 0 and y_true[i] != y_pred[i]:
                FN += 1
    else:
        for i in range(len(y_pred)):
            if y_true[i] == y_pred[i] == 1:
                TP += 1
            if y_pred[i] == 1 and y_true[i] != y_pred[i]:
                FP += 1
            if y_true[i] == y_pred[i] == 0:
                TN += 1
            if y_pred[i] == 0 and y_true[i] != y_pred[i]:
                FN += 1

    results['TP'] = TP
    results['FP'] = FP
    results['TN'] = TN
    results['FN'] = FN
    results['sensitivity/recall'] = TP / ((TP + FN) if (TP + FN) > 0 else 1)
    results['specificity'] = TN / ((TN + FP) if (TN + FP) > 0 else 1)
    results['precision'] = TP / ((TP + FP) if (TP + FP) > 0 else 1)
    results['accuracy'] = (TP + TN) / (TP + FN + TN + FP)
    results['f1'] = 2 * TP / (2 * TP + FP + FN)

    return results


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    # else:
        # print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.4f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

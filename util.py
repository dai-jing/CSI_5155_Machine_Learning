import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import display
from scipy.stats import ttest_rel
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedShuffleSplit


def plot_roc_curve(roc_y_true, roc_y_score, algorithm: str, target_as_readmission=True):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(roc_y_true==(i+1), roc_y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    colors = ["aqua", "darkorange", "cornflowerblue"]

    for i in range(3):
        if i == 0:
            if target_as_readmission:
                plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label='readmitted: <30 curve (area = %0.2f)' % roc_auc[i])
            else:
                plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label='age: young curve (area = %0.2f)' % roc_auc[i])
        elif i == 1:
            if target_as_readmission:
                plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label='readmitted: >30 curve (area = %0.2f)' % roc_auc[i])
            else:
                plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label='age: mid-age curve (area = %0.2f)' % roc_auc[i])
        else:
            if target_as_readmission:
                plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label='readmitted: NO curve (area = %0.2f)' % roc_auc[i])
            else:
                plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label='age: old curve (area = %0.2f)' % roc_auc[i])

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for {}'.format(algorithm))
    plt.legend(loc="lower right")
    plt.show()


def semi_supervised_learning_splits(X, y, test_size):
    _, test_index = next(StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42).split(X, y))

    y_copy = np.copy(y)
    y_copy[test_index] = -1

    return y_copy


def self_training_splits(X, y, test_size):
    labelled_index, unlabelled_index = next(StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42).split(X, y))

    X_labelled = X[labelled_index]
    y_labelled = y[labelled_index]
    X_unlabelled = X[unlabelled_index]

    return X_labelled, y_labelled, X_unlabelled


def model_scores(scores):
    additional_scores = (scores.mean(), scores.std())
    results = np.append(scores, additional_scores).reshape(-1, 1)

    return results


def paired_t_tests_scores(diffs, pvalue):
    additional_differences = (diffs.mean(), diffs.std(), pvalue)
    results = np.append(diffs, additional_differences).reshape(-1, 1)

    return results


def display_cv_f1_scores(scores: list, algorithms: list):
    row_headers = ["1", "2", "3", "4", "5", "avg", "stdev"]

    table_df = pd.DataFrame(data=np.concatenate(scores, axis=1), columns=algorithms, index=row_headers)

    display(table_df)


def display_statistical_differences(scores: list, algorithms: list):
    paired_t_tests = []
    paired_t_tests_column_headers = []
    paired_t_tests_row_headers = ["1", "2", "3", "4", "5", "avg", "stdev", "p-value"]

    algorithm_scores = list()
    for i in range(len(scores)):
        algorithm_scores.append({"name": algorithms[i], "scores": scores[i]})

    for i in range(len(algorithm_scores)):
        for j in range(i + 1, len(algorithm_scores)):
            algorithm_pairs = algorithm_scores[i].get("name") + "-" + algorithm_scores[j].get("name")
            paired_t_tests_column_headers.append(algorithm_pairs)
            first_scores = algorithm_scores[i].get("scores")
            second_scores = algorithm_scores[j].get("scores")
            differences = []
            for z in range(5):
                first_score = first_scores[z]
                second_score = second_scores[z]

                differences.append(first_score - second_score)

            _, p_value = ttest_rel(first_scores, second_scores)
            paired_t_tests.append(paired_t_tests_scores(np.array(differences), p_value[0]))

    paired_t_tests_table_df = pd.DataFrame(data=np.concatenate(paired_t_tests, axis=1),
                                           columns=paired_t_tests_column_headers, index=paired_t_tests_row_headers)

    display(paired_t_tests_table_df)

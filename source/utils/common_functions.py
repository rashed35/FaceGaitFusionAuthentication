"""
Created by Sanjay at 4/10/2020

Feature: Enter feature name here
Enter feature description here
"""
import decimal

from sklearn import metrics


def report_results(actual, predicted, cmat_flag=False, rprt_flag=False):
    """
    Reports results from the actual labels and predicted labels
    :param actual: original labels (test_y)
    :param predicted: predicted labels (pred_y)
    :param cmat_flag: Confusion Matrix flag
    :param rprt_flag: Detailed Report flag
    :return:
    """
    score = metrics.accuracy_score(actual, predicted)
    print('\t Accuracy: %.4f' % score)
    conf_matrix, report = None, None
    if cmat_flag:
        conf_matrix = metrics.confusion_matrix(actual, predicted)
        print('\t Confusion Matrix:')
        print(conf_matrix)
    if rprt_flag:
        report = metrics.classification_report(actual, predicted)
        print('\t Report')
        print(report)
    print()
    return score, conf_matrix, report

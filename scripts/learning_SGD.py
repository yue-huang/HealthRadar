# Test script on part of the data: python3 learning_SGD.py train.csv test.csv
# Usage on all data : python3 learning_SGD.py LLCP2015.csv_preprocessing.csv_transformed_train.csv LLCP2015.csv_preprocessing.csv_transformed_test.csv

# Train model on training set with cross-validation using grid search for best parameters.
# Evaluate model - Bias/variance: Validation curve, learning curve.
# Evaluate model - quantify quality of evaluations: Precision, recall, f1 score, PR curve, ROC curve.

import multiprocessing

multiprocessing.set_start_method('forkserver')

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import model_selection
import time
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import RFE
import pickle


def get_X_Y_from_csv(csv_name):
    df = pd.read_csv(csv_name)
    array = df.values
    X = array[:, 1:]
    Y = array[:, 0]
    return X, Y


def estimator_separate_train_cv_test(train, cv, test):
    X, Y = get_X_Y_from_csv(train)
    n = X.shape[0]
    Xcv, Ycv = get_X_Y_from_csv(cv)
    Xtest, Ytest = get_X_Y_from_csv(test)
    alpha = np.logspace(-6, -1, 15)
    n_iter = np.ceil(10 ** 6 / n)
    best_f1_score = 0
    best_alpha = 1
    for a in alpha:
        clf = SGDClassifier(loss="log", alpha=a, class_weight='balanced')
        clf.fit(X, Y)
        Ycv_predict = clf.predict(Xcv)
        newscore = metrics.f1_score(Ycv, Ycv_predict)
        if newscore > best_f1_score:
            s = pickle.dumps(clf)
            best_alpha = a
            best_f1_score = newscore
    best_clf = pickle.loads(s)
    print('best alpha is', best_alpha)
    print('f1 score on train data:', metrics.f1_score(Y, best_clf.predict(X)))
    print('f1 score on cv data:', best_f1_score)
    print('f1 score on test data:', metrics.f1_score(Ytest, best_clf.predict(Xtest)))
    probas_ = best_clf.predict_proba(Xtest)


def plot_precision_recall(Ytest, Yprobas, Ypredict):
    precision, recall, thresholds = metrics.precision_recall_curve(Ytest, Yprobas)
    PRarea = metrics.auc(recall, precision)
    print("Area Under PR Curve: %0.2f" % PRarea)
    print('f1 score:', metrics.f1_score(Ytest, Ypredict))
    print('Precision score:', metrics.precision_score(Ytest, Ypredict))
    print('Recall score:', metrics.recall_score(Ytest, Ypredict))
    plt.subplot(211)
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall: AUC=%0.2f' % PRarea)
    plt.legend(loc="lower left")
    return plt


def plot_ROC(Ytest, Yprobas):
    fpr, tpr, thresholds = metrics.roc_curve(Ytest, Yprobas)
    ROCarea = metrics.auc(fpr, tpr)
    print("Area under ROC curve: %.2f" % ROCarea)
    plt.subplot(212)
    plt.plot(fpr, tpr, label='roc curve')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('ROC curve: AUC=%0.2f' % ROCarea)
    plt.legend(loc="lower left")
    return plt


def plot_validation_curve(X, Y):
    param_range = np.logspace(-6, -1, 10)
    train_scores, test_scores = model_selection.validation_curve(SGDClassifier(loss='log', class_weight='balanced'), X,
                                                                 Y,
                                                                 param_name="alpha", param_range=param_range,
                                                                 scoring='f1', n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.title("Validation Curve")
    plt.xlabel("alpha")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")


def plot_learning_curve(X, Y):
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, test_scores = model_selection.learning_curve(
        SGDClassifier(loss='log', alpha=0.0003, class_weight='balanced'), X, Y, scoring='f1', n_jobs=2,
        train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def grid_search(train, test):
    param_grid = [
        {'estimator__alpha': np.logspace(-6, -1, 10), 'estimator__loss': ['log']},
        {'estimator__alpha': np.logspace(-6, -1, 10), 'estimator__loss': ['hinge']},
    ]
    est = SGDClassifier(class_weight='balanced')
    selector = RFE(est, n_features_to_select=40)
    clf = model_selection.GridSearchCV(selector, param_grid=param_grid, scoring='f1', n_jobs=1, error_score=0)
    X, Y = get_X_Y_from_csv(train)
    Xtest, Ytest = get_X_Y_from_csv(test)
    clf.fit(X, Y)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    Ypredict = clf.predict(Xtest)
    print(metrics.classification_report(Ytest, Ypredict))
    print()

    if clf.best_estimator_.estimator.get_params()['loss'] == 'log':
        probas_ = clf.predict_proba(Xtest)
        plot_precision_recall(Ytest, probas_[:, 1], Ypredict)


def main(train, test):
    start = time.time()
    grid_search(train, test)
    print('It took %.2f seconds.' % (time.time() - start))
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

# Test script on part of the data: python3 ../scripts/learning_LR.py train.csv test.csv
# Usage on all data : python3 ../scripts/learning_LR.py LLCP2015.csv_preprocessing.csv_transformed_train.csv LLCP2015.csv_preprocessing.csv_transformed_test.csv

# Train model on training set with cross-validation using grid search for best parameters.
# Evaluate model - Bias/variance: Validation curve, learning curve.
# Evaluate model - quantify quality of evaluations: Precision, recall, f1 score, PR curve.



import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import model_selection
import time
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

def get_X_Y_from_csv(csv_name):
    df = pd.read_csv(csv_name)
    array = df.values
    X = array[:, 1:]
    Y = array[:, 0]
    return X, Y


def estimator_separate_train_test(train, test):
    X, Y = get_X_Y_from_csv(train)
    n = X.shape[0]
    Xtest, Ytest = get_X_Y_from_csv(test)
    est = LogisticRegressionCV(cv=5, scoring='f1', class_weight='balanced')
    est.fit(X, Y)
    print('best C is', est.C_)
    Ypredict = est.predict(Xtest)
    probas_ = est.predict_proba(Xtest)
    plot_precision_recall(Ytest, probas_[:, 1], Ypredict)

def grid_search(train, test):
    param_grid = [
        {'estimator__C': np.logspace(-4, 4, 10), 'estimator__class_weight':[{0: w, 1:(1-w)} for w in [0.1, 0.2, 0.4, 0.6, 0.9]]}
    ]
    est = LogisticRegression()
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
    Ypredict = clf.predict(Xtest)
    print(metrics.classification_report(Ytest, Ypredict))
    print()
    probas_ = clf.predict_proba(Xtest)
    plot_precision_recall(Ytest, probas_[:, 1], Ypredict)

def plot_precision_recall(Ytest, Yprobas, Ypredict):
    precision, recall, thresholds = metrics.precision_recall_curve(Ytest, Yprobas)
    PRarea = metrics.auc(recall, precision)
    print("Area Under PR Curve: %0.2f" % PRarea)
    print('f1 score:', metrics.f1_score(Ytest, Ypredict))
    print('Precision:', metrics.precision_score(Ytest, Ypredict))
    print('Recall:', metrics.recall_score(Ytest, Ypredict))
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall: AUC=%0.2f' % PRarea)
    plt.legend(loc="lower left")
    return plt


def plot_validation_curve(train):
    X, Y = get_X_Y_from_csv(train)
    param_range = np.logspace(-4, 4, 10)
    train_scores, test_scores = model_selection.validation_curve(
        LogisticRegression(class_weight='balanced', solver='lbfgs', n_jobs=1), X,
        Y,
        param_name="C", param_range=param_range,
        scoring='f1', n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.title("Validation Curve")
    plt.xlabel("C")
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


def plot_learning_curve(train):
    X, Y = get_X_Y_from_csv(train)
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, test_scores = model_selection.learning_curve(
        LogisticRegression(C=0.0464, class_weight='balanced'), X, Y, scoring='f1', n_jobs=1,
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


def main(train, test):
    start = time.time()
    estimator_separate_train_test(train, test)
    #grid_search(train,test) #too slow
    #plot_validation_curve(train)
    #plot_learning_curve(train)
    print('It took %.2f seconds.' % (time.time() - start))
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

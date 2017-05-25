# Test script on part of the data: python3 ../scripts/learning_kernel.py train.csv test.csv
# Usage on all data : python3 ../scripts/learning_kernel.py LLCP2015.csv_preprocessing.csv_transformed_train.csv LLCP2015.csv_preprocessing.csv_transformed_test.csv

# Train model on training set with cross-validation using grid search for best parameters.


import pandas as pd
import numpy as np
import sys

# http://stackoverflow.com/questions/4931376/generating-matplotlib-graphs-without-a-running-x-server
# fig.savefig('temp.png')
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import model_selection
import time
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline

# https://github.com/scikit-learn/scikit-learn/issues/6943#issuecomment-229003483
from sklearn.externals.joblib import parallel

parallel.MIN_IDEAL_BATCH_DURATION = 1.
parallel.MAX_IDEAL_BATCH_DURATION = parallel.MIN_IDEAL_BATCH_DURATION * 10


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
    rbf_feature = RBFSampler()
    X_features = rbf_feature.fit_transform(X)
    clf = SGDClassifier(class_weight="balanced")
    clf.fit(X_features, Y)
    Ypredict = clf.predict(X_features)

    Xtest_features = rbf_feature.transform(Xtest)
    Ytest_predict = clf.predict(Xtest_features)

    print('score on training and test sets: {:.2f} {:.2f}'.format(clf.score(X_features, Y),
                                                                  clf.score(Xtest_features, Ytest)))
    print('f1 score on training and test sets:{:.2f} {:.2f}'.format(metrics.f1_score(Y, Ypredict),
                                                                    metrics.f1_score(Ytest, Ytest_predict)))
    print('Precision on training and test sets:{:.2f} {:.2f}'.format(metrics.precision_score(Y, Ypredict),
                                                                     metrics.precision_score(Ytest, Ytest_predict)))
    print('Recall on training and test sets: {:.2f} {:.2f}'.format(metrics.recall_score(Y, Ypredict),
                                                                   metrics.recall_score(Ytest, Ytest_predict)))


def grid_search(train, test):
    param_grid = {'rbf__gamma': np.logspace(-4, 3, 7), 'rbf__n_components': [200, 1000, 5000],
                  'sgd__class_weight': [None, 'balanced'], 'sgd__alpha': np.logspace(-6, -1, 5)}
    # Note: 'balanced' performs better than 'None', which yields model with f1 = 0.
    rbf = RBFSampler()
    sgd = SGDClassifier(verbose=100)
    pipe = Pipeline(steps=[('rbf', rbf), ('sgd', sgd)])
    clf = model_selection.GridSearchCV(pipe, param_grid=param_grid, scoring='f1', n_jobs=1, error_score='raise',
                                       verbose=100)
    X, Y = get_X_Y_from_csv(train)
    Xtest, Ytest = get_X_Y_from_csv(test)
    clf.fit(X, Y)

    # print best results and stats for each value of rbf__n_components.
    results = clf.cv_results_
    bestresults = dict()
    for i in range(len(results['param_rbf__n_components'])):
        n = results['param_rbf__n_components'][i]
        if n not in bestresults.keys():
            bestresults[n] = [results['mean_test_score'][i], results['std_test_score'][i],
                              results['mean_train_score'][i], results['mean_fit_time'][i], results['params'][i]]
        elif results['mean_test_score'][i] > bestresults[n][0]:
            bestresults[n] = [results['mean_test_score'][i], results['std_test_score'][i],
                              results['mean_train_score'][i], results['mean_fit_time'][i], results['params'][i]]

    print("Best parameters found for each n on development set:")
    for n in bestresults.keys():
        print('f1 score on test and train set: {:.2f} (+/-{:.2f}), {:.2f}, time: {:.2f} seconds, for {!r}'.format(
            bestresults[n][0], bestresults[n][1], bestresults[n][2], bestresults[n][3], bestresults[n][4]))
    print()
    print("Detailed classification report:")
    print()
    Ypredict = clf.predict(Xtest)
    print(metrics.classification_report(Ytest, Ypredict))


def main(train, test):
    start = time.time()
    # estimator_separate_train_test(train, test)
    grid_search(train, test)
    print('It took %.2f seconds.' % (time.time() - start))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

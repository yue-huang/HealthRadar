# Test script on part of the data: python3 ../scripts/learning_kernel.py train.csv test.csv
# Usage on all data : python3 ../scripts/learning_kernel.py LLCP2015.csv_preprocessing.csv_transformed_train.csv LLCP2015.csv_preprocessing.csv_transformed_test.csv

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
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline


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

    print('score on training and test sets: {:.2f} {:.2f}'.format(clf.score(X_features, Y), clf.score(Xtest_features, Ytest)))
    print('f1 score on training and test sets:{:.2f} {:.2f}'.format(metrics.f1_score(Y, Ypredict),metrics.f1_score(Ytest, Ytest_predict)))
    print('Precision on training and test sets:{:.2f} {:.2f}'.format(metrics.precision_score(Y, Ypredict),metrics.precision_score(Ytest, Ytest_predict)))
    print('Recall on training and test sets: {:.2f} {:.2f}'.format(metrics.recall_score(Y, Ypredict),metrics.recall_score(Ytest, Ytest_predict)))

    #est = LogisticRegressionCV(cv=5, scoring='f1', class_weight='balanced')
    #est.fit(X, Y)
    #print('best C is', est.C_)
    #Ypredict = est.predict(Xtest)
    #probas_ = est.predict_proba(Xtest)
    #plot_precision_recall(Ytest, probas_[:, 1], Ypredict)

def grid_search(train, test):
    param_grid = {'rbf__gamma': np.logspace(-4, 3, 7), 'rbf__n_components': [200,1000,5000,10000],
         'sgd__class_weight':[None, 'balanced'], 'sgd__alpha':np.logspace(-6, -1, 5)}

    # C_range = np.logspace(-2, 10, 13)
    #gamma_range = np.logspace(-9, 3, 13)
    #For an initial search, a logarithmic grid with basis 10 is often helpful. Using a basis of 2, a finer tuning can be achieved but at a much higher cost.
    rbf = RBFSampler()
    sgd = SGDClassifier()
    pipe = Pipeline(steps=[('rbf', rbf), ('sgd', sgd)])
    clf = model_selection.GridSearchCV(pipe, param_grid=param_grid, scoring='f1', n_jobs=1, error_score=0)
    X, Y = get_X_Y_from_csv(train)
    Xtest, Ytest = get_X_Y_from_csv(test)
    clf.fit(X, Y)

    #Ypredict = clf.predict(X)
    #Ytest_predict = clf.predict(Xtest)

    #print('score on training and test sets: {:.2f} {:.2f}'.format(clf.score(X_features, Y), clf.score(Xtest_features, Ytest)))
    #print('f1 score on training and test sets:{:.2f} {:.2f}'.format(metrics.f1_score(Y, Ypredict),metrics.f1_score(Ytest, Ytest_predict)))
    #print('Precision on training and test sets:{:.2f} {:.2f}'.format(metrics.precision_score(Y, Ypredict),metrics.precision_score(Ytest, Ytest_predict)))
    #print('Recall on training and test sets: {:.2f} {:.2f}'.format(metrics.recall_score(Y, Ypredict),metrics.recall_score(Ytest, Ytest_predict)))

    results = clf.cv_results_
    bestresults = dict()
    for i in range(len(results['param_rbf__n_components'])):
        n = results['param_rbf__n_components'][i]
        if n not in bestresults.keys():
            bestresults[n] = [results['mean_test_score'][i], results['std_test_score'][i],results['mean_train_score'][i], results['mean_fit_time'][i],results['params'][i]]
        elif results['mean_test_score'][i]>bestresults[n][0]:
            bestresults[n] = [results['mean_test_score'][i], results['std_test_score'][i],results['mean_train_score'][i], results['mean_fit_time'][i],results['params'][i]]

    print("Best parameters found for each n on development set:")
    for n in bestresults.keys():
        print('f1 score on test and train set: {:.2f} (+/-{:.2f}), {:.2f}, time: {:.2f} seconds, for {!r}'.format(bestresults[n][0],bestresults[n][1],bestresults[n][2],bestresults[n][3],bestresults[n][4]))
    print()
    print("Detailed classification report:")
    print()
    Ypredict = clf.predict(Xtest)
    print(metrics.classification_report(Ytest, Ypredict))


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
    #estimator_separate_train_test(train, test)
    grid_search(train,test)
    #plot_validation_curve(train)
    #plot_learning_curve(train)
    print('It took %.2f seconds.' % (time.time() - start))
    #plt.show()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

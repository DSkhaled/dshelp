from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import StratifiedKFold

from collections import defaultdict


def validation(model, X, y, type, cv=5, seed=0):
    """
    :param model: model to evaluate
    :param X: features
    :param y: labels
    :param type: type of evaluation
    :param cv: number of folds
    :return: return the usual scores for variety of methods
    cross validation
    cross validation time series
    cross validation
    """
    scoring = ["accuracy", "precision", "recall", "f1"]
    scoring_fun = {"accuracy": accuracy_score,
                   "f1_score": f1_score,
                   "precision_score": precision_score,
                   "recall_score": recall_score}

    if type == 'cv':
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

        print_scores(scoring, cv_scores)

    if type == "ts":
        ts_scores = defaultdict(list)
        tscv = TimeSeriesSplit(n_splits=cv)

        for train, test in tscv.split(X):

            X_train, X_test, y_train, y_test = X.loc[train], X.loc[test], y.loc[train], y.loc[test]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            for metric in scoring:
                ts_scores[metric].append(scoring_fun[metric](y_pred, y_test))

        print_scores(scoring, ts_scores)

    if type == "ss":
        ss_scores = defaultdict(list)
        skf = StratifiedKFold(n_splits=cv)

        for train, test in skf.split(X, y):

            X_train, X_test, y_train, y_test = X.loc[train], X.loc[test], y.loc[train], y.loc[test]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            for metric in scoring:
                ss_scores[metric].append(scoring_fun[metric](y_pred, y_test))

            print_scores(scoring, ss_scores)


def print_scores(scoring, dict_scores):
    for metric in scoring:
        print(text_result(metric, dict_scores[metric], 2 * dict_scores[metric]))


def text_result(metric, mean, std):
    return metric.capitalize() + ": " + "{mean:0.2f} (+/- {confidence:0.2f})".format(mean=mean.mean(),
                                                                                     confidence=2 * std.std())
import numpy as np


def use_gem(classifier, X):
    n = X.shape[0]
    d = X.shape[1]
    Y = np.zeros((n, 1))

    for i in range(n):
        found = 0
        j = 1

        while found == 0:
            #t = classifier[j][0]
            #t1 = X[i,:]
            #t2 = np.subtract(classifier[j][0], X[i,:])
            #t3 = np.linalg.norm(np.subtract(classifier[j][0], X[i,:]), ord=2)
            #t4 = classifier[j][1]
            if np.linalg.norm(np.subtract(classifier[j][0], X[i,:]), ord=2) < classifier[j][1]:
                found = 1
                Y[i] = classifier[j][2]
            else:
                j += 1

    return Y


def use_gem_ensemble(classifiers, X, boosted=False):
    preds = []

    if boosted:
        for classifier in classifiers:
            preds.append(use_gem(classifier[0], X[:, classifier[1]]))
    else:
        for classifier in classifiers:
            preds.append(use_gem(classifier, X))

    num_classifiers = len(classifiers)
    np.asarray(preds)

    return np.around(np.multiply(np.sum(preds, axis=0), 1/num_classifiers))

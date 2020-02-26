import train_gem_balls
import use_gem_balls
import validate_gem_balls
import numpy as np
from sklearn.metrics import confusion_matrix
import tqdm


basic_gem_res = []
for i in tqdm(range(100)):
    N=200
    Xtrain = np.subtract(np.random.rand(N, 5), 0.5)
    Ytrain = 1*np.array(np.sum(Xtrain, axis=1) > 0)

    train_results = train_gem_balls.train_gem(Xtrain, Ytrain)
    classifier = train_results[0]
    Xtest = np.subtract(np.random.rand(10000, 5), 0.5)
    Ytest_true = 1 * np.array(np.sum(Xtest, axis=1) > 0)
    Ytest_pred = use_gem_balls.use_gem(classifier, Xtest)

    tn, fp, fn, tp = confusion_matrix(Ytest_pred, Ytest_true).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    acc = (tp + tn) / (tp + fp + fn + tn)

    basic_gem_res.append([fpr, fnr, acc])

np.savetxt("basic_gem_res.csv", np.asarray(basic_gem_res), delimiter=',')

print("checkpoint1")

bagged_gem_res = []
for i in tqdm(range(100)):
    N=200
    Xtrain = np.subtract(np.random.rand(N, 5), 0.5)
    Ytrain = 1*np.array(np.sum(Xtrain, axis=1) > 0)

    ensemble = train_gem_balls.create_gem_ensemble(Xtrain, Ytrain, per_data=0.5, num_gem=75)
    Xtest = np.subtract(np.random.rand(10000, 5), 0.5)
    Ytest_true = 1 * np.array(np.sum(Xtest, axis=1) > 0)
    Ytest_pred_bagged = use_gem_balls.use_gem_ensemble(ensemble, Xtest)

    tn, fp, fn, tp = confusion_matrix(Ytest_pred_bagged, Ytest_true).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    acc = (tp + tn) / (tp + fp + fn + tn)

    bagged_gem_res.append([fpr, fnr, acc])

np.savetxt("bagged_gem_res.csv", np.asarray(bagged_gem_res), delimiter=',')

print("checkpoint2")

boosted_gem_res = []
for i in tqdm(range(100)):
    N=200
    Xtrain = np.subtract(np.random.rand(N, 5), 0.5)
    Ytrain = 1*np.array(np.sum(Xtrain, axis=1) > 0)

    ensemble = train_gem_balls.create_gem_ensemble(Xtrain, Ytrain, mode="boosted",
                                                   per_data=0.5, per_features=0.8, num_gem=75)
    Xtest = np.subtract(np.random.rand(10000, 5), 0.5)
    Ytest_true = 1 * np.array(np.sum(Xtest, axis=1) > 0)
    Ytest_pred_boosted = use_gem_balls.use_gem_ensemble(ensemble, Xtest, boosted=True)

    tn, fp, fn, tp = confusion_matrix(Ytest_pred_boosted, Ytest_true).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    acc = (tp + tn) / (tp + fp + fn + tn)

    boosted_gem_res.append([fpr, fnr, acc])

np.savetxt("boosted_gem_res.csv", np.asarray(boosted_gem_res), delimiter=',')

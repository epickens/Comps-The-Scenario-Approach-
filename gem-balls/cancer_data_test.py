import train_gem_balls
import use_gem_balls
import validate_gem_balls
import numpy as np
from sklearn.metrics import confusion_matrix
import loader


Xtrain, Xtest, Ytrain, Ytest = loader.load_data('data/names.csv', 'data/breast-cancer-wisconsin.data',
                                                    cols=True, is_wisconsin_bc=True)
"""
train_results = train_gem_balls.train_gem(Xtrain, Ytrain)
classifier = train_results[0]
kp = train_results[1]
kn = train_results[2]
Np = train_results[3]
Nn = train_results[4]

LOOFalseNegative = kp / Np
LOOFalsePositive = kn / Nn

beta = 0.05
valid_results = validate_gem_balls.validate_gem(kp,kn,Np,Nn,beta,beta)

Ytest_pred = use_gem_balls.use_gem(classifier, Xtest)

tn, fp, fn, tp = confusion_matrix(Ytest_pred, Ytest).ravel()
print(fp/(fp+tn))
print(fn/(fn+tp))
print((tp+tn)/(tp+fp+fn+tn))
"""

ensemble = train_gem_balls.create_gem_ensemble(Xtrain, Ytrain, mode="boosted", per_data=0.5, per_features=0.8, num_gem=10)

Ytest_pred_bagged = use_gem_balls.use_gem_ensemble(ensemble, Xtest, boosted=True)

tn, fp, fn, tp = confusion_matrix(Ytest_pred_bagged, Ytest).ravel()
print(fp/(fp+tn))
print(fn/(fn+tp))
print((tp+tn)/(tp+fp+fn+tn))

print("done")
exit()
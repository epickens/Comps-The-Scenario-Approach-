import train_gem_balls
import use_gem_balls
import validate_gem_balls
import numpy as np
from sklearn.metrics import confusion_matrix


N=200

Xtrain = np.subtract(np.random.rand(N, 5), 0.5)
Ytrain = 1*np.array(np.sum(Xtrain, axis=1) > 0)

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

G_false_negative_rate = valid_results[0]
G_false_positive_rate = valid_results[1]

Xtest = np.subtract(np.random.rand(10000, 5), 0.5)
Ytest_true = 1*np.array(np.sum(Xtest, axis=1) > 0)
Ytest_pred = use_gem_balls.use_gem(classifier, Xtest)

tn, fp, fn, tp = confusion_matrix(Ytest_pred, Ytest_true).ravel()
print(fp/(fp+tn))
print(fn/(fn+tp))
print((tp+tn)/(tp+fp+fn+tn))
#false_neg_sim = len(np.argwhere(Ytest_true==1 and Ytest_pred==0))/len(np.argwhere(Ytest_true==1))
#false_pos_sim = len(np.argwhere(Ytest_true==0 and Ytest_pred==1))/len(np.argwhere(Ytest_true==0))

print("done")
exit()

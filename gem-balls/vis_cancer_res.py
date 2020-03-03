import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


basic = pd.read_csv("basic_gem_cancer.csv")
bagged = pd.read_csv("bagged_gem_cancer.csv")
boosted = pd.read_csv("boosted_gem_cancer.csv")

sns.distplot(bagged.values[:,2], bins=30)
plt.axvline(x=np.mean(bagged.values[:,2]), c='r', linestyle="--", label="Mean Accuracy")
plt.legend()
plt.show()

sns.distplot(boosted.values[:,2], bins=30)
plt.axvline(x=np.mean(boosted.values[:,2]), c='r', linestyle="--", label="Mean Accuracy")
plt.legend()
plt.show()

#sns.distplot(basic.values[:,2], bins=30, hist=False, label="Basic")
sns.distplot(bagged.values[:,2], bins=30, hist=False, label="Bagged")
sns.distplot(boosted.values[:,2], bins=30, hist=False, label="Random Forest")
plt.axvline(x=basic.values[0,2], linestyle="--", label="Baseline Accuracy")
plt.title("Accuracy by Method")
plt.legend()
plt.show()

sns.distplot(bagged.values[:,1], bins=30, hist=False, label="Bagged")
sns.distplot(boosted.values[:,1], bins=30, hist=False, label="Random Forest")
plt.axvline(x=basic.values[0,1], linestyle="--", label="Baseline FNR")
plt.legend()
plt.title("False Negative Rate by Method")
plt.show()

sns.distplot(bagged.values[:,0], bins=30, hist=False, label="Bagged")
sns.distplot(boosted.values[:,0], bins=30, hist=False, label="Random Forest")
plt.axvline(x=basic.values[0,0], linestyle="--", label="Baseline FPR")
plt.legend()
plt.title("False Positive Rate by Method")
plt.show()

print("Mean bagged acc: {}".format(np.mean(bagged.values[:,2])))
print("Mean random forest acc: {}".format(np.mean(boosted.values[:,2])))
plt.show()
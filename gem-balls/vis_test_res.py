import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


basic = pd.read_csv("basic_gem_res.csv")
bagged = pd.read_csv("bagged_gem_res.csv")
boosted = pd.read_csv("boosted_gem_res.csv")

sns.distplot(basic.values[:,2], bins=30)
plt.axvline(x=np.mean(basic.values[:,2]), c='r', linestyle="--", label="Mean Accuracy")
plt.legend()
plt.show()

sns.distplot(bagged.values[:,2], bins=30)
plt.axvline(x=np.mean(bagged.values[:,2]), c='r', linestyle="--", label="Mean Accuracy")
plt.legend()
plt.show()

sns.distplot(boosted.values[:,2], bins=30)
plt.axvline(x=np.mean(boosted.values[:,2]), c='r', linestyle="--", label="Mean Accuracy")
plt.legend()
plt.show()

sns.distplot(basic.values[:,2], bins=30, hist=False, label="Basic")
sns.distplot(bagged.values[:,2], bins=30, hist=False, label="Bagged")
sns.distplot(boosted.values[:,2], bins=30, hist=False, label="Random Forest")
plt.axvline(x=np.mean(basic.values[:,2]), linestyle="--",
            color="tab:blue", label="Mean Basic Accuracy")
plt.axvline(x=np.mean(bagged.values[:,2]), linestyle="--",
            color="tab:orange", label="Mean Bagged Accuracy")
plt.axvline(x=np.mean(boosted.values[:,2]), linestyle="--",
            color="tab:green", label="Mean Random Forest Accuracy")
plt.title("Accuracy by Method")
plt.legend()
plt.show()

x = np.random.rand(200, 1)
y = np.concatenate([np.multiply(np.power(x, 2), -1) + np.random.uniform(-0.1, 0.1, 1)
                    for x in x])

#data = zip(x, y)

plt.scatter(x,y)#,color="#a50026")
plt.plot(np.linspace(0, 1, 100), np.multiply(np.power(np.linspace(0, 1, 100), 2), -1),
         color='orange', linewidth=4, label="$y=-x^2$")
plt.plot(np.linspace(0, 1, 100), np.multiply(np.power(np.linspace(0, 1, 100), 2), -1)-0.1,
         color='orange', linewidth=2, linestyle="--")
plt.plot(np.linspace(0, 1, 100), np.multiply(np.power(np.linspace(0, 1, 100), 2), -1)+0.1,
         color='orange', linewidth=2, linestyle="--")
plt.legend()
plt.show()
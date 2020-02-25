import train_gem_balls
import use_gem_balls
import validate_gem_balls
import numpy as np
import matplotlib.pyplot as plt


N = 200

Xtrain = np.subtract(np.random.rand(N, 2), 0.5)
Ytrain = 1*np.array(np.sum(Xtrain, axis=1) > 0)

#plt.style.use('grayscale')
plt.scatter(Xtrain[:,0], Xtrain[:,1], c=Ytrain, cmap='RdYlBu')
plt.show()

train_results = train_gem_balls.train_gem(Xtrain, Ytrain)
classifier = train_results[0]

circles = []
for i in reversed(range(len(classifier))):
    if classifier[i][2] == 0:
        circles.append(plt.Circle(classifier[i][0], classifier[i][1], color='#a50026', alpha=0.5))
    else:
        circles.append(plt.Circle(classifier[i][0], classifier[i][1], color='w', alpha=0.5)) # #313695


fig, ax = plt.subplots()
for circle in circles:
    ax.add_artist(circle)
#plt.scatter(Xtrain[:,0], Xtrain[:,1], c=Ytrain, cmap='RdYlBu')
plt.xlim(-0.5, 0.5)
plt.ylim(-0.5, 0.5)
plt.show()
print("done")
exit()
#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
#%%
# generate dataset for classification
X, y = make_classification(
    n_samples=200, n_features=2, n_redundant=0, n_clusters_per_class=1,
    n_classes=2, class_sep=0.8, random_state=42
)

h = .02 # grid step size
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#%%
k_values = [1, 5, 20] # the number of neighbors

plt.figure(figsize=(15, 5))
for i, k in enumerate(k_values):
    clf = KNeighborsClassifier(n_neighbors=k) # KNN Algorithm
    clf.fit(X, y) # Empirical Risk Minimization
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # Predict on test data
    Z = Z.reshape(xx.shape)
    
    plt.subplot(1, len(k_values), i+1)
    plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='gray')
    plt.title(f'K={k}', fontsize=20)
    plt.contour(xx, yy, Z, levels=[0.5], linewidths=3, colors='k') # decision boundary
plt.tight_layout()
plt.show()
plt.close()
#%%
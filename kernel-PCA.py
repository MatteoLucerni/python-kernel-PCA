import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.decomposition import KernelPCA


def plot_boundary(model, X, Y, labels=["Classe 0", "Classe 1"], figsize=(12, 10)):

    plt.figure(figsize=figsize)

    h = 0.02

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    X_m = X[Y == 1]
    X_b = X[Y == 0]
    plt.scatter(X_b[:, 0], X_b[:, 1], c="green", edgecolor="white", label=labels[0])
    plt.scatter(X_m[:, 0], X_m[:, 1], c="red", edgecolor="white", label=labels[1])
    plt.legend()
    plt.show()


X, Y = make_circles(n_samples=1000, noise=0.1, factor=0.2, random_state=1)
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.show()

# without kernel PCA
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

lr = LogisticRegression()
lr.fit(X_train, Y_train)
pred = lr.predict(X_test)
pred_train = lr.predict(X_train)

acc = accuracy_score(Y_test, pred)
acc_train = accuracy_score(Y_train, pred_train)

print(f"Test: {acc} / Train {acc_train}")

plot_boundary(lr, X, Y)


# kernel PCA
kpca = KernelPCA(kernel="rbf", gamma=5)

kpc = kpca.fit_transform(X)

plt.scatter(kpc[:, 0], kpc[:, 1], c=Y)
plt.show()

# 1 dimension
plt.scatter(kpc[:, 0], np.zeros((1000, 1)), c=Y)
plt.show()

fpc = kpc[:, 0]
fpc = fpc.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(fpc, Y, random_state=0)

lr = LogisticRegression()
lr.fit(X_train, Y_train)
pred = lr.predict(X_test)
pred_train = lr.predict(X_train)

acc = accuracy_score(Y_test, pred)
acc_train = accuracy_score(Y_train, pred_train)

print(f"K-PCA -> Test: {acc} / Train {acc_train}")

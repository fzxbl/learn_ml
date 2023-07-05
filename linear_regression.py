import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class MyLinearRegression:
    """
    损失函数 = 每个样本损失的平方和，而每个样本的损失(xw-y)^2在w参数空间内必然是处处可微的，因此梯度必然存在。如果损失函数关于w参数空间不可微，则无法使用梯度下降法求解
    线性回归其实可以使用正规解形式，但是要求X.T*X是满秩的，或者正定的，实际上样本数量可能与特征数量相差较大，大部分时候不是满秩的，梯度下降法更适用
    使用两种梯度下降法来进行线性回归：批量梯度下降、随机小批量梯度下降
    """
    def __init__(self) -> None:
        self._center = 0
        self._std = 0
        self._weight = None

    def fit(self, X:np.ndarray, y:np.ndarray, learning_rate, loss_change, max_iter):
        X = self._rescale(X)
        X = np.hstack((X, np.ones((X.shape[0],1))))
        w = np.zeros((X.shape[1],1))
        last_loss = 0
        iters = 0
        for i in range(max_iter):
            grad = X.T @ (X @ w - y)
            loss = ((y - X @ w).T @ (y - X @ w))[0][0]
            if np.abs(loss - last_loss) < loss_change:
                break
            w -= learning_rate * grad
            last_loss = loss
            iters += 1
        self._weight = w
        return w, last_loss, iters
    
    def fit2(self, X:np.ndarray, y:np.ndarray, learning_rate, loss_change, epoch, batch_size):
        X = self._rescale(X)
        X = np.hstack((X, np.ones((X.shape[0],1))))
        w = np.zeros((X.shape[1],1))
        
        last_loss = 0
        iters = 0
        for i in range(epoch):
            # 乱序
            indices = np.random.permutation(X.shape[0])
            X = X[indices]
            y = y[indices]
            for i in range(0, batch_size, X.shape[0]):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                grad = X_batch.T @ (X_batch @ w - y_batch)
                loss = ((y_batch - X_batch @ w).T @ (y_batch - X_batch @ w))[0][0]
                if np.abs(loss - last_loss) < loss_change:
                    break
                w -= learning_rate * grad
                last_loss = loss
                iters += 1
        self._weight = w
        return w, last_loss, iters

    def predict(self, X):
        X = (X - self._center) / self._std
        X = np.hstack((X, np.ones((X.shape[0],1))))
        y = X @ self._weight
        return y

    def _rescale(self, X:np.ndarray):
        self._center = X.mean(axis=0)
        self._std = X.std(axis=0)
        X = (X - self._center) / self._std
        return X

if __name__ == '__main__': 
    
    data = load_diabetes()
    X = data.data
    y = data.target

    # 作图可以发现各个维度和最终结果的相关性较差,线性回归解决不了该问题
    X_fig = X.T
    index = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
    fig, axs = plt.subplots(5, 2)
    for i in range(0, X_fig.shape[0], 2):
        row = int(i / 2 )  
        print(row)
        axs[row][0].scatter(X_fig[i], y)
        axs[row][0].set_xlabel(index[i])
        axs[row][0].set_yticks(np.arange(0, 300, 50))

        axs[row][1].scatter(X_fig[i+1], y)
        axs[row][1].set_xlabel(index[i+1])
        axs[row][1].set_yticks(np.arange(0, 300, 50))
        
    fig.set_size_inches(h=12, w=10)
    # plt.subplots_adjust(hspace=6)
    plt.tight_layout()
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    y_train = y_train.reshape((y_train.shape[0],1))
    y_test = y_test.reshape((y_test.shape[0],1))

    l1 = MyLinearRegression()
    w, loss, iters = l1.fit(X_train, y_train, learning_rate=0.001, loss_change=0.0000000001, max_iter=30000)
    # w, loss, iters = l1.fit2(X_train, y_train, learning_rate=0.001, loss_change=0.0000000001, epoch=500, batch_size=40)
    print('权重1', w)
    y_pre1 = l1.predict(X_test)
    mse1 = mean_squared_error(y_test, y_pre1)
    print('均方误差1：', mse1)

    l2 = LinearRegression()
    l2.fit(X_train, y_train)
    print('权重2', l2.coef_)
    y_pre2 = l2.predict(X_test)
    mse2 = mean_squared_error(y_test, y_pre2)
    print('均方误差2：', mse2)

    print('预测结果对比', np.hstack((y_pre1, y_pre2)))

    print('预测值-真实值', np.hstack((y_pre1, y_test))) 
    
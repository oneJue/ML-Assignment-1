import numpy as np


class Model:
    def __init__(self):
        """
        初始化线性回归模型
        """
        self.weights = None
        self.bias = None

    def predict(self, X):
        """
        预测样本的值

        Args:
            X: 输入特征，形状为 (n_samples, n_features) 的numpy数组

        Returns:
            numpy.ndarray: 形状为 (n_samples,) 的预测值数组
        """
        return np.dot(X, self.weights) + self.bias
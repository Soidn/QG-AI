import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'SimHei'


dataset = pd.read_csv(r'C:\Users\24378\Downloads\housing.csv')

# print(dataset.isnull().sum())
# print(dataset.duplicated().sum())
# print(dataset.describe())

# pd.set_option('display.max_columns', None)
# print(dataset.head())

dME = dataset['MEDV']
# print(medv_range.min(), medv_range.max())
medv_distrib, bin_edges = np.histogram(dME,range = [5,50],bins=9)

bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

plt.figure(figsize=(12,6))
interval_width = bin_edges[1] - bin_edges[0]
#这行代码通过获取 bin_edges 中第二个元素（索引为 1）和第一个元素（索引为 0），计算它们的差值，从而得出数据分组中第一个区间的宽度。
plt.bar(bin_centers,medv_distrib, width=interval_width, color="skyblue",label="房价分布",alpha=0.8)
plt.xticks(bin_edges)
plt.title('房价在各区间的分布')
plt.ylabel('频数')
plt.xlabel('房价区间')
plt.legend()

corr = dataset.corr()['MEDV'].sort_values(ascending=False)
# print(corr)

dRM = dataset['RM']
dLS = dataset['LSTAT']
plt.figure(figsize=(24,6))

sd1 =plt.subplot(1,2,1)
sd1.scatter(dRM,dME,marker='o',color='r')
sd1.set_title('平均每居民房数与房价关系散点图')

sd2 = plt.subplot(1,2,2)
sd2.scatter(dLS,dME,marker='^',color='b')
sd2.set_title('人口中地位较低人群的百分数与房价关系散点图')


# SOS
# 实在是学不过来了，诚实相告: 以下内容百分之98借助了AI之力，仅供个人今后学习参考
# 自定义数据分割函数
# 特征标准化




class LinearRegressionOLS:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        self.coef_ = np.linalg.pinv(X.T @ X) @ X.T @ y

    def predict(self, X):
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.coef_


class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iters=1000, fit_intercept=True):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.fit_intercept = fit_intercept
        self.coef_ = None

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)

        for _ in range(self.n_iters):
            y_pred = X @ self.coef_
            gradient = (2 / n_samples) * X.T @ (y_pred - y)
            self.coef_ -= self.lr * gradient

    def predict(self, X):
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.coef_


# 自定义标准化函数
class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return (X - self.mean) / self.std

    def transform(self, X):
        return (X - self.mean) / self.std

# 自定义评估指标
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_num = int(len(X) * test_size)
    return (X[indices[:-test_num]], X[indices[-test_num:]],
            y[indices[:-test_num]], y[indices[-test_num:]])

X = dataset.drop('MEDV', axis=1).values.astype(float)  # 特征矩阵
y = dataset['MEDV'].values.astype(float)               # 目标变量

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练和评估OLS模型
ols = LinearRegressionOLS(fit_intercept=True)
ols.fit(X_train_scaled, y_train)
y_pred_ols = ols.predict(X_test_scaled)

# 训练和评估GD模型
gd = LinearRegressionGD(learning_rate=0.1, n_iters=10000, fit_intercept=True)
gd.fit(X_train_scaled, y_train)
y_pred_gd = gd.predict(X_test_scaled)

# 可视化结果对比
plt.figure(figsize=(12, 6))

# OLS结果可视化
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_ols, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('真实房价')
plt.ylabel('预测房价')
plt.title('OLS预测结果')

# GD结果可视化
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_gd, alpha=0.6, color='orange')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('真实房价')
plt.ylabel('预测房价')
plt.title('梯度下降预测结果')

plt.tight_layout()
plt.show()

# 打印评估指标
print("OLS结果:")
print(f"MSE: {mean_squared_error(y_test, y_pred_ols):.2f}")
print(f"R²: {r2_score(y_test, y_pred_ols):.2f}\n")

print("梯度下降结果:")
print(f"MSE: {mean_squared_error(y_test, y_pred_gd):.2f}")
print(f"R²: {r2_score(y_test, y_pred_gd):.2f}")
#  谢谢AI，下周再努力吧

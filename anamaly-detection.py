import numpy as np
from matplotlib import pyplot as plt
from scipy import io as spio
import pandas

'''异常检测主运行程序'''
def anomalyDetection_example():
    '''加载并显示数据'''
    dataframe = pandas.read_table('analyse-20180325.txt', delim_whitespace=True)
    print(dataframe.shape)
    print(dataframe.head(5))
    dataset = dataframe.values
    X = dataset.astype('float32')

    #data = spio.loadmat('data1.mat')
    #X = data['X']
    print('float_data\' shape is (%d %d) :' % (X.shape) )
    #for x in X:
        #print('real value is : %s' %  X[0])

    '''多元高斯分布函数，并可视化拟合的边界'''
    # 参数估计（求均值和方差）
    mu, sigma2 = estimateGaussian(X)
    print (mu,sigma2)


    p = multivariateGaussian(X, mu, sigma2)
    print (p)


    '''选择异常点（在交叉验证CV上训练得到最好的epsilon）
    Xval = data['Xval']
    yval = data['yval']  # y=1代表异常
    pval = multivariateGaussian(Xval, mu, sigma2)  # 计算CV上的概率密度值
    epsilon, F1 = selectThreshold(yval, pval)  # 选择最优的epsilon临界值
    print(u'在CV上得到的最好的epsilon是：%e' % epsilon)
    print(u'对应的F1Score值为：%f' % F1)
    outliers = np.where(p < epsilon)  # 找到小于临界值的异常点，并作图
    print(outliers)
    '''

# 参数估计函数（就是求均值和方差）
def estimateGaussian(X):
    m, n = X.shape
    mu = np.zeros((n, 1))
    sigma2 = np.zeros((n, 1))

    mu = np.mean(X, axis=0)  # axis=0表示列，每列的均值
    sigma2 = np.var(X, axis=0)  # 求每列的方差
    return mu, sigma2


# 多元高斯分布函数
def multivariateGaussian(X, mu, Sigma2):
    k = len(mu)
    if (Sigma2.shape[0] > 1):
        Sigma2 = np.diag(Sigma2)
    '''多元高斯分布函数'''
    X = X - mu
    argu = (2 * np.pi) ** (-k / 2) * np.linalg.det(Sigma2) ** (-0.5)
    p = argu * np.exp(-0.5 * np.sum(np.dot(X, np.linalg.inv(Sigma2)) * X, axis=1))  # axis表示每行
    return p

# 选择最优的epsilon，即：使F1Score最大
def selectThreshold(yval, pval):
    '''初始化所需变量'''
    bestEpsilon = 0.
    bestF1 = 0.
    F1 = 0.
    step = (np.max(pval) - np.min(pval)) / 1000
    '''计算'''
    for epsilon in np.arange(np.min(pval), np.max(pval), step):
        cvPrecision = pval < epsilon
        tp = np.sum((cvPrecision == 1) & (yval == 1).ravel()).astype(float)  # sum求和是int型的，需要转为float
        fp = np.sum((cvPrecision == 1) & (yval == 0).ravel()).astype(float)
        fn = np.sum((cvPrecision == 0) & (yval == 1).ravel()).astype(float)
        precision = tp / (tp + fp)  # 精准度
        recision = tp / (tp + fn)  # 召回率
        F1 = (2 * precision * recision) / (precision + recision)  # F1Score计算公式
        if F1 > bestF1:  # 修改最优的F1 Score
            bestF1 = F1
            bestEpsilon = epsilon
    return bestEpsilon, bestF1


if __name__ == '__main__':
    anomalyDetection_example()

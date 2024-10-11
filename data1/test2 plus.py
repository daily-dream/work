import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.covariance import EmpiricalCovariance

# 读取男性和女性的身高和体重数据
male_data = pd.read_csv('./MALE.TXT', sep=r'\s+')
female_data = pd.read_csv('./FEMALE.TXT', sep=r'\s+')

# 提取身高和体重特征
male_features = male_data.iloc[:, :2].values
female_features = female_data.iloc[:, :2].values

# 计算均值和协方差
mean_male = np.mean(male_features, axis=0)
cov_male = EmpiricalCovariance().fit(male_features).covariance_
mean_female = np.mean(female_features, axis=0)
cov_female = EmpiricalCovariance().fit(female_features).covariance_

# 定义多变量正态分布
def multivariate_normal_male(x):
    return st.multivariate_normal.pdf(x, mean=mean_male, cov=cov_male)

def multivariate_normal_female(x):
    return st.multivariate_normal.pdf(x, mean=mean_female, cov=cov_female)

# 设置先验概率
p_male = 0.9
p_female = 0.1

# 定义损失矩阵
loss_matrix = np.array([[0, 2], [2, 0]])

# 定义最小风险决策函数
def min_risk_decision(x, p_male_func, p_female_func):
    # 计算每个决策的预期风险
    p_male_x = p_male_func(x)
    p_female_x = p_female_func(x)
    risk_if_male = loss_matrix[0, 0] * p_male_x + loss_matrix[0, 1] * p_female_x
    risk_if_female = loss_matrix[1, 0] * p_male_x + loss_matrix[1, 1] * p_female_x
    # 选择预期风险最小的决策
    if risk_if_male < risk_if_female:
        return 1  # 预测为男性
    else:
        return 2  # 预测为女性

# 定义后验概率函数
def p_male_x(x):
    return (multivariate_normal_male(x) * p_male) / (multivariate_normal_male(x) * p_male + multivariate_normal_female(x) * p_female)

def p_female_x(x):
    return (multivariate_normal_female(x) * p_female) / (multivariate_normal_male(x) * p_male + multivariate_normal_female(x) * p_female)

# 读取测试数据
test_data = pd.read_csv('./test2.txt', sep=r'\s+')
features = test_data.iloc[:, :2].values
labels_test = test_data.iloc[:, 2].values

# 预测标签
label_pred = [min_risk_decision(x, p_male_x, p_female_x) for x in features]

# 计算错误率
error_rate = sum(label_pred[i] != labels_test[i] for i in range(len(labels_test))) / len(labels_test)

print('错误率:', error_rate)

# 比较不同先验概率的影响
prior_combinations = [(0.5, 0.5), (0.75, 0.25), (0.9, 0.1)]
for p_male, p_female in prior_combinations:
    label_pred = [min_risk_decision(x, lambda x: (multivariate_normal_male(x) * p_male) / (multivariate_normal_male(x) * p_male + multivariate_normal_female(x) * p_female),
                                    lambda x: (multivariate_normal_female(x) * p_female) / (multivariate_normal_male(x) * p_male + multivariate_normal_female(x) * p_female)) for x in features]
    error_rate = sum(label_pred[i] != labels_test[i] for i in range(len(labels_test))) / len(labels_test)
    print(f'先验概率 {p_male} vs. {p_female} 的错误率:', error_rate)
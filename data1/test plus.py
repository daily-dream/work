# -*- coding: utf-8 -*-
# Jonathan.K.Wolf
# 2022/5/12

import numpy as np
import pandas as pd
import scipy.stats as st
import math

# 读取男性和女性的身高数据，使用 sep='\s+' 替代 delim_whitespace
male_data = pd.read_csv('./MALE.TXT', sep='\s+')
female_data = pd.read_csv('./FEMALE.TXT', sep='\s+')

# 只取身高数据
male_height = male_data.iloc[:, 0].values
female_height = female_data.iloc[:, 0].values

# 计算男性和女性的身高均值和方差
theta1_male = np.mean(male_height)
theta2_male = np.var(male_height)
theta1_female = np.mean(female_height)
theta2_female = np.var(female_height)

print('male mean:{}, variance:{}'.format(theta1_male, theta2_male))
print('female mean:{}, variance:{}'.format(theta1_female, theta2_female))


# 定义正态分布的概率密度函数
def normal_male(x):
    return st.norm.pdf(x, loc=theta1_male, scale=math.sqrt(theta2_male))


def normal_female(x):
    return st.norm.pdf(x, loc=theta1_female, scale=math.sqrt(theta2_female))


# 设置先验概率
p_male = 0.5
p_female = 0.5


# 定义后验概率函数
def p_male_x(x):
    return (normal_male(x) * p_male) / (normal_male(x) * p_male + normal_female(x) * p_female)


def p_female_x(x):
    return (normal_female(x) * p_female) / (normal_male(x) * p_male + normal_female(x) * p_female)


# 读取测试数据，使用 sep='\s+' 替代 delim_whitespace
test_data = pd.read_csv('./test2.txt', sep='\s+')
height = test_data.iloc[:, 0].values
labels_test = test_data.iloc[:, 2].values

# 根据性别标签将测试数据分为男性和女性
male_height_test = []
female_height_test = []
label_test = []

for i in range(len(labels_test)):
    if labels_test[i] == 1:
        male_height_test.append(height[i])
        label_test.append(1)
    elif labels_test[i] == 2:
        female_height_test.append(height[i])
        label_test.append(2)


# 定义最小风险决策函数
def min_risk_decision(x, a, b):
    # 计算损失
    loss_if_male = a * p_female_x(x)  # 分类为男性但实际为女性时的损失
    loss_if_female = b * p_male_x(x)  # 分类为女性但实际为男性时的损失

    # 比较损失，选择损失较小的分类
    if loss_if_male < loss_if_female:
        return 1  # 男性
    else:
        return 2  # 女性


# 假设错误分类的代价相同
a = 1
b = 18

# 预测标签
label_pred_min_risk = []
for index in (male_height_test + female_height_test):
    label_pred_min_risk.append(min_risk_decision(index, a, b))

# 计算正确率
correct_predictions = sum(1 for actual, predicted in zip(label_test, label_pred_min_risk) if actual == predicted)
acc_min_risk = correct_predictions / len(label_test) if label_test else 0
acc_min_risk = round(acc_min_risk, 2)

# 输出结果
print('实际标签:', label_test)
print('预测标签:', label_pred_min_risk)
print('在测试集上使用最小风险Bayes决策的预测正确率：{:.2f}%'.format(acc_min_risk * 100 if acc_min_risk != 0 else 0))
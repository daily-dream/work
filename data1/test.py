

import numpy as np
import pandas as pd
import scipy.stats as st
import math

# 读取男性和女性的身高数据，最开始使用 delim_whitespace，结果报错了，使用使用 sep='\s+' 替代
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

# 读取测试数据
test_data = pd.read_csv('./test1.txt', sep='\s+')
height = test_data.iloc[:, 0].values
labels_test = test_data.iloc[:, 2].values

# 根据性别标签将测试数据分为男性和女性
male_height = []
female_height = []
label_test = []

for i in range(len(labels_test)):
    if labels_test[i] == 1:
        male_height.append(height[i])
        label_test.append(1)
    elif labels_test[i] == 2:
        female_height.append(height[i])
        label_test.append(2)

# 预测标签
label_pred = []
for index in (male_height + female_height):
    if p_male_x(index) > p_female_x(index):
        label_pred.append(1)
    else:
        label_pred.append(2)

# 计算男生预测正确率
acc_of_male = sum(1 for index_1 in male_height if p_male_x(index_1) > p_female_x(index_1)) / len(male_height) if male_height else 0
acc_of_male = round(acc_of_male, 2)

# 计算女生预测正确率
acc_of_female = sum(1 for index_2 in female_height if p_male_x(index_2) < p_female_x(index_2)) / len(female_height) if female_height else 0
acc_of_female = round(acc_of_female, 2)

# 输出结果
print('实际标签:', label_test)
print('预测标签:', label_pred)
print('在测试集上预测男生的正确率：{:.2f}%'.format(acc_of_male * 100 if acc_of_male != 0 else 0))
print('在测试集上预测女生的正确率：{:.2f}%'.format(acc_of_female * 100 if acc_of_female != 0 else 0))
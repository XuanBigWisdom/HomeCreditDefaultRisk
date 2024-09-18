# -*- coding: utf-8 -*-            
# @Author : Maxiaoxuan
# @Time : 2024/5/25 17:10
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 加载数据
data_path = 'selected_features_data1.csv'  # 请确保这是正确的文件路径
data = pd.read_csv(data_path)
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))  # 设置图形的大小，可根据实际需求调整
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

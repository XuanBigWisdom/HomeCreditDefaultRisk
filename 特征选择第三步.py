# -*- coding: utf-8 -*-            
# @Author : Maxiaoxuan
# @Time : 2024/5/25 17:14
# vif.py
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 加载数据
data_path = 'selected_features_data1.csv'
data = pd.read_csv(data_path)

# 准备数据，确保数据无缺失值
data = data.dropna()

# 计算VIF值
vif_data = pd.DataFrame()
vif_data["feature"] = data.columns
vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]

# 显示VIF值大于10的特征
print(vif_data[vif_data["VIF"] > 10])

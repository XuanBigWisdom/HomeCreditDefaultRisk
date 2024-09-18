# -*- coding: utf-8 -*-            
# @Author : Maxiaoxuan
# @Time : 2024/5/21 22:29
import pandas as pd

# 加载数据
data_path = 'application_train.csv'  # 请确保这是正确的文件路径
data = pd.read_csv(data_path)

# 计算每个特征的缺失值比例
missing_percentage = data.isnull().mean() * 100

# 筛选掉缺失值比例大于80%的特征
features_with_less_missing = data.columns[missing_percentage < 30]
reduced_data = data[features_with_less_missing]

# 计算方差
variance = reduced_data.var()

# 假设我们选择方差的阈值为0.01（根据具体情况调整）
features_with_higher_variance = variance[variance > 0.01].index
final_data = reduced_data[features_with_higher_variance]

# 将最终筛选后的特征保存到CSV文件中
final_data.to_csv('selected_features_data1.csv', index=False)

# 打印结果
print(f"原始特征数量: {data.shape[1]}")
print(f"移除缺失值过多的特征后的数量: {reduced_data.shape[1]}")
print(f"基于方差筛选后的特征数量: {final_data.shape[1]}")

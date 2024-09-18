# -*- coding: utf-8 -*-            
# @Author : Maxiaoxuan
# @Time : 2024/5/25 17:17
# lgbm_feature_importance.py
# lgbm_feature_importance_enhanced.py
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
data_path = 'selected_features_data1.csv'
data = pd.read_csv(data_path)

# 定义特征和目标
X = data.drop('TARGET', axis=1)  # 假设数据中有一个名为'target'的列
y = data['TARGET']

# 创建LightGBM模型
model = lgb.LGBMClassifier()
model.fit(X, y)

# 获取特征重要性并创建DataFrame
importances = model.feature_importances_
feature_names = X.columns
feature_imports = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_imports.sort_values(by='Importance', ascending=False, inplace=True)

# 将特征重要性输出到CSV文件
feature_imports.to_csv('feature_importance.csv', index=False)

# 绘制特征重要性图
plt.figure(figsize=(12, 8))
sns.barplot(data=feature_imports, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importances by LightGBM')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()  # 确保图标元素不会重叠
plt.show()

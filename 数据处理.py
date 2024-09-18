# -*- coding: utf-8 -*-            
# @Author : Maxiaoxuan
# @Time : 2024/5/21 1:20
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import ColumnTransformer

def calculate_iv(X, y):
    # 这里用一个简单的逻辑回归作为例子来模拟IV值计算过程
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    if X.shape[1] == 0:
        return 0  # 防止空特征的情况
    model = LogisticRegression()
    model.fit(X, y)
    predictions = model.predict_proba(X)[:, 1]
    return roc_auc_score(y, predictions)

# 加载数据
data_path = 'application_train.csv'  # 修改为你的数据文件路径
data = pd.read_csv(data_path)

# 填充数值型缺失值为-9999
numeric_columns = data.select_dtypes(include=['number']).columns
data[numeric_columns] = data[numeric_columns].fillna(-9999)

# 转换非数值型特征为数值型（独热编码）
categorical_columns = data.select_dtypes(exclude=['number']).columns
data = pd.get_dummies(data, columns=categorical_columns)

# 计算每个特征的缺失率
missing_rates = data.isin([-9999]).mean()

# 选取缺失率小于80%的特征
selected_features = missing_rates[missing_rates < 0.8].index.tolist()

# 应用VarianceThreshold来剔除低方差的特征，这里我们暂时假设低方差对应IV值不合适
selector = VarianceThreshold(threshold=0.02)  # 假设阈值
data_reduced = data[selected_features]
data_reduced_transformed = selector.fit_transform(data_reduced)

# 将结果转换回DataFrame
data_final = pd.DataFrame(data_reduced_transformed, columns=data_reduced.columns[selector.get_support()])

# 假设'y'是目标变量列名
target_column = 'TARGET'
y = data[target_column] if target_column in data.columns else None

# 计算选定特征的IV值，并筛选IV值合适的特征
if y is not None:
    iv_values = {col: calculate_iv(data_final[[col]], y) for col in data_final.columns}
    # 筛选IV值合适的特征，这里以IV值大于0.1为例
    final_features = [col for col, iv in iv_values.items() if iv > 0.1]
    data_final = data_final[final_features]

# 保存处理后的数据
output_path = 'application_train_processed_data.csv'
data_final.to_csv(output_path, index=False)
print(f"Processed data saved to {output_path}")

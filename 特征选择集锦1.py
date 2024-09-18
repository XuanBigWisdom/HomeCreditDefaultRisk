# -*- coding: utf-8 -*-            
# @Author : Maxiaoxuan
# @Time : 2024/5/26 0:50
# comprehensive_feature_selection.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.feature_selection import RFE, chi2, SelectKBest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.api import add_constant

# 数据加载
data_path = 'selected_features_data1.csv'
data = pd.read_csv(data_path)
X = data.drop('TARGET', axis=1)
y = data['TARGET']


import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.api import add_constant

def calculate_vif(X):
    X = X.dropna()  # 删除任何包含缺失值的行
    X_const = add_constant(X)
    vifs = pd.DataFrame()
    vifs["VIF Factor"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
    vifs["Feature"] = X_const.columns
    # 返回VIF结果，除去常数项
    return vifs[vifs["Feature"] != "const"]


rf = RandomForestClassifier()
rf.fit(X, y)
feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
feature_importances.sort_values(ascending=False).to_csv('random_forest_importances.csv')
plt.figure(figsize=(10, 8))
feature_importances.sort_values().plot(kind='barh', color='skyblue')
plt.title('Random Forest Feature Importances')
plt.savefig('random_forest_importances.png')

lasso = LassoCV(cv=5).fit(X, y)
lasso_importance = pd.Series(np.abs(lasso.coef_), index=X.columns)
lasso_importance.sort_values(ascending=False).to_csv('lasso_coefficients.csv')
plt.figure(figsize=(10, 8))
lasso_importance.sort_values().plot(kind='barh', color='green')
plt.title('Lasso Coefficients')
plt.savefig('lasso_coefficients.png')

# RFE
logistic = LogisticRegression()
selector = RFE(logistic, n_features_to_select=10, step=1)
selector.fit(X, y)
rfe_support = pd.Series(selector.support_, index=X.columns)
rfe_support.to_csv('rfe_support.csv')
plt.figure(figsize=(10, 8))
rfe_support.sort_values().plot(kind='barh', color='red')
plt.title('RFE Feature Selection')
plt.savefig('rfe_feature_selection.png')


# 卡方检验
chi_scores, _ = chi2(X, y)
chi_scores = pd.Series(chi_scores, index=X.columns)
chi_scores.sort_values(ascending=False).to_csv('chi2_scores.csv')
plt.figure(figsize=(10, 8))
chi_scores.sort_values().plot(kind='barh', color='orange')
plt.title('Chi-squared Scores')
plt.savefig('chi_squared_scores.png')

# PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(12, 10))
sns.heatmap(pd.DataFrame(X_pca).corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('PCA Component Correlations')
plt.savefig('pca_correlation_matrix.png')

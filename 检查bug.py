# -*- coding: utf-8 -*-            
# @Author : Maxiaoxuan
# @Time : 2024/5/21 22:18
import pandas as pd
import re

# 加载数据
data_path = 'application_train_processed_data.csv'  # 修改为你的数据文件路径
data = pd.read_csv(data_path)

# 列出所有包含特殊字符的列名
special_char_columns = []
regex = re.compile(r'[^\w]')  # 匹配任何非字母数字下划线的字符

for col in data.columns:
    if regex.search(col):  # 如果列名中有特殊字符
        special_char_columns.append(col)

# 打印包含特殊字符的列名
if special_char_columns:
    print("Columns with special characters:")
    for col in special_char_columns:
        print(col)
else:
    print("No columns with special characters found.")

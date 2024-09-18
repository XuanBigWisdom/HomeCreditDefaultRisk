# -*- coding: utf-8 -*-            
# @Author : Maxiaoxuan
# @Time : 2024/5/20 17:23
import pandas as pd
from googletrans import Translator
import time

def translate_text(text, src='en', dest='zh-cn'):
    translator = Translator()
    try:
        translation = translator.translate(text, src=src, dest=dest)
        return translation.text
    except Exception as e:
        print(f"Error during translation: {e}")
        return text  # 在出错时返回原文

def translate_csv(file_path, columns, retries=3):
    df = pd.read_csv(file_path)
    for column in columns:
        if column in df.columns:
            print(f"Translating column: {column}")
            for i in range(len(df[column])):
                if pd.notnull(df[column].iloc[i]):
                    retry_count = 0
                    while retry_count < retries:
                        result = translate_text(df[column].iloc[i])
                        if result != df[column].iloc[i]:
                            df.at[i, column] = result
                            break
                        retry_count += 1
                        time.sleep(1)  # 稍等一秒再重试
        else:
            print(f"Column {column} not found in the CSV.")
    df.to_csv(file_path, index=False)
    print("Translation completed and file saved.")

translate_csv('HomeCredit_columns_description.csv', ['Table','Row','Description','Special'])



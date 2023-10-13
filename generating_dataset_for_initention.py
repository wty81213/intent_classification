import os 
os.chdir(r"c:\\Users\\cloudy822\\Desktop\\intent classification")
import pandas as pd 
from sklearn import preprocessing

intent_data = pd.read_excel('./source/dataset/raw_data/intention/訓練資料_全部資料暨初版提供_v4.xlsx',sheet_name = 1,engine = 'openpyxl')
intent_data = intent_data[['大類','小類','內容']]

problem_table = pd.read_excel('./source/dataset/raw_data/intention/附件一、資料格式_問題類別清單及回覆內容_晧揚回饋版本.xlsx',sheet_name = 0,engine = 'openpyxl')
for c in ['問題大類類別','問題大類','大類回應方式','大類回應方式.1']:
    problem_table[c] = problem_table[c].fillna(method = 'ffill')
problem_table = problem_table[['問題大類類別','問題大類','問題編號','問題類別']]
problem_table = problem_table[problem_table['問題編號'].notnull()]

### renaming for columns 
intent_data = intent_data.rename(columns = {'大類':'problem_large_name',\
                                                '小類':'problem_name',\
                                                '內容':'sentence'})
problem_table = problem_table.rename(columns = {'問題大類類別':'problem_large_id',\
                                                '問題大類':'problem_large_name',\
                                                '問題編號':'problem_id',
                                                '問題類別':'problem_name'})


intent_data = intent_data.merge(problem_table[['problem_large_id','problem_large_name']].drop_duplicates(),on = 'problem_large_name',how = 'left')
intent_data = intent_data.merge(problem_table[['problem_id','problem_name']].drop_duplicates(),on = 'problem_name',how = 'left')
intent_data = intent_data[['problem_large_id','problem_large_name','problem_id','problem_name','sentence']]


intent_data.to_csv('./source/dataset/original_data/dialogue_data_20230816.csv',index = False)
intent_data.to_excel('./source/dataset/original_data/dialogue_data_20230816.xlsx',index = False)


LE1 = preprocessing.LabelEncoder().fit(intent_data['problem_large_id'])
le1_name_mapping = dict(zip(LE1.classes_, LE1.transform(LE1.classes_)))
print(le1_name_mapping)
LE2 = preprocessing.LabelEncoder().fit(intent_data['problem_id'])
le2_name_mapping = dict(zip(LE2.classes_, LE2.transform(LE2.classes_)))
print(le2_name_mapping)
print(len(le2_name_mapping))
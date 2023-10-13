import os 
os.chdir(r"c:\\Users\\cloudy822\\Desktop\\intent classification")
import pandas as pd 
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit


problem_table = pd.read_excel('./source/dataset/raw_data/intention/附件一、資料格式_問題類別清單及回覆內容_晧揚回饋版本.xlsx',sheet_name = 0,engine = 'openpyxl')
for c in ['問題大類類別','問題大類','大類回應方式','大類回應方式.1']:
    problem_table[c] = problem_table[c].fillna(method = 'ffill')
problem_table = problem_table[['問題大類類別','問題大類','問題編號','問題類別']]
problem_table = problem_table[problem_table['問題編號'].notnull()]
problem_table = problem_table.rename(columns = {'問題大類類別':'problem_large_id',\
                                                '問題大類':'problem_large_name',\
                                                '問題編號':'problem_id',
                                                '問題類別':'problem_name'})

intent_data = pd.read_excel('./source/dataset/raw_data/intention/20230820_訓練資料_版3_生成資料.xlsx',sheet_name = 1,engine = 'openpyxl')
intent_data = intent_data.rename(columns = {'大類':'problem_large_name',\
                                                '小類':'problem_name',\
                                                '內容':'sentence',
                                                'TAG':'source'})
intent_data = intent_data[intent_data['是否不易判別'].isnull()]
intent_data = intent_data.merge(problem_table[['problem_large_id','problem_large_name']].drop_duplicates(),on = 'problem_large_name',how = 'left')
intent_data = intent_data.merge(problem_table[['problem_id','problem_name']].drop_duplicates(),on = 'problem_name',how = 'left')
intent_data = intent_data[['problem_large_id','problem_large_name','problem_id','problem_name','sentence','source']]
intent_data = intent_data.reset_index(drop=True)
print('The shape of intent_data = {}'.format(intent_data.shape))


intent_data = intent_data.reset_index()

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)

sub_intent_data = intent_data[intent_data['source'].isin(['非調整', '調整'])]
train_index, test_index = list(split.split(sub_intent_data,sub_intent_data['problem_id']))[0]


selected_index = sub_intent_data.iloc[test_index]['index'].tolist() 
print(len(selected_index))


valid_data = intent_data[intent_data['index'].isin(selected_index)]
print('The shape of intent_data = {}'.format(valid_data.shape))

train_data = intent_data[~intent_data.index.isin(selected_index)]
print('The shape of intent_data = {}'.format(train_data.shape))


print('tain_data = {} | {} \n {}'.format(train_data['problem_name'].nunique(),train_data.shape[0],train_data['problem_name'].value_counts()))
print('test_data = {} | {} \n {}'.format(valid_data['problem_name'].nunique(),valid_data.shape[0],valid_data['problem_name'].value_counts()))

train_data.drop(columns = ['source','index']).to_csv('./source/dataset/original_data/dialogue_data_20230917_v1.csv',index = False)
# train_data.drop(columns = 'source','index').to_excel('./source/dataset/original_data/dialogue_data_20230917_v1.xlsx',index = False)
valid_data.drop(columns = ['source','index']).to_csv('./source/dataset/original_data/dialogue_test_data_20230917_v1.csv',index = False)
# valid_data.drop(columns = 'source','index').to_excel('./source/dataset/original_data/dialogue_test_data_20230917_v1.xlsx',index = False)



#-----------------------------------------------
LE1 = preprocessing.LabelEncoder().fit(intent_data['problem_large_id'])
le1_name_mapping = dict(zip(LE1.classes_, LE1.transform(LE1.classes_)))
print(le1_name_mapping)
LE2 = preprocessing.LabelEncoder().fit(intent_data['problem_id'])
le2_name_mapping = dict(zip(LE2.classes_, LE2.transform(LE2.classes_)))
print(le2_name_mapping)
print(len(le2_name_mapping))
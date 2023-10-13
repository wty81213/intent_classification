import os 
os.chdir(r"c:\\Users\\cloudy822\\Desktop\\intent classification")
import pandas as pd 
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit

intent_data = pd.read_excel('./source/dataset/raw_data/intention/dialogue_dataset_20230915.xlsx',sheet_name = 'Sheet1',engine = 'openpyxl')
intent_data = intent_data[['problem_large_id','problem_large_name','problem_id','problem_name','sentence','source']]


class_table = intent_data[['problem_large_id','problem_large_name','problem_id','problem_name']].\
    groupby(['problem_large_id','problem_large_name','problem_id','problem_name'],as_index=False).size().sort_values(by = 'problem_id')
print(class_table.shape)

intent_data = intent_data.reset_index()

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.15, random_state = 42)

sub_intent_datav1 = intent_data[intent_data['source'].isin(['非調整', '調整'])]
train_index1, valid_index1 = list(split.split(sub_intent_datav1,sub_intent_datav1['problem_id']))[0]

sub_intent_datav2 = intent_data[intent_data['source'].isin(['chatGPT'])]
train_index2, valid_index2 = list(split.split(sub_intent_datav2,sub_intent_datav2['problem_id']))[0]

selected_index = sub_intent_datav1.iloc[valid_index1]['index'].tolist() + sub_intent_datav2.iloc[valid_index2]['index'].tolist()
print(len(selected_index))

valid_data = intent_data[intent_data['index'].isin(selected_index)]
train_data = intent_data[~intent_data.index.isin(selected_index)]

print('all_data = {} | {} \n {}'.format(intent_data['problem_name'].nunique(),intent_data.shape[0],intent_data['problem_name'].value_counts()))
print('tain_data = {} | {} \n {}'.format(train_data['problem_name'].nunique(),train_data.shape[0],train_data['problem_name'].value_counts()))
print('valid_data = {} | {} \n {}'.format(valid_data['problem_name'].nunique(),valid_data.shape[0],valid_data['problem_name'].value_counts()))

train_data.drop(columns = ['source','index']).to_csv('./source/dataset/original_data/dialogue_dataset_20230915.csv',index = False)
valid_data.drop(columns = ['source','index']).to_csv('./source/dataset/original_data/dialogue_valid_dataset_20230915.csv',index = False)



# intent_data.to_csv('./source/dataset/original_data/dialogue_dataset_20230914.csv',index = False)


# 第一版 file_path=./source/dataset/original_data/dialogue_data_20230821.csv
# 第二版 file_path=./source/dataset/original_data/dialogue_dataset_20230914.csv
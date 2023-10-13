import os
import pandas as pd 
from tqdm import tqdm
from data_loader import SentimentDataset
from model import BertForSequenceClassifier

import torch
from torch.utils.data import DataLoader


def problem_prediction(sentence, model, device, mapping_table):
    
    if isinstance(sentence,str):
        sentence = [sentence]
    else :
        assert isinstance(sentence,list)
    
    model = model.to(device)
    
    senti_data  = (input_sentence,[0]*len(input_sentence))
    input_dataset = SentimentDataset(senti_data, token_path = token_path)
    dataloader = DataLoader(input_dataset, batch_size=2, collate_fn=input_dataset.collate_fn)

    total_logits = []
    total_target = []
    batchs_iterator = tqdm(dataloader, unit = 'batch', position = 0)  
    for batch_idx, (_, data, target) in enumerate(batchs_iterator):
        data = {k:v.to(device) for k,v in data.items()}
        target = target.to(device)
        with torch.no_grad():
            output = model(**data, return_dict = True)  
        total_logits.append(output.logits)
        total_target.append(target)

        batchs_iterator.set_description('Prediction')

    if device == 'cpu':
        total_logits = torch.cat(total_logits, dim = 0)
        pred_labels = torch.argmax(total_logits, dim=1).numpy()
    else:        
        total_logits = torch.cat(total_logits, dim = 0).to('cpu')
        pred_labels = torch.argmax(total_logits, dim=1).to('cpu').numpy()

    mapping_table_swap = {v: k for k, v in mapping_table.items()}
    pred_labels = [mapping_table_swap[i] for i in pred_labels]

    return pred_labels

if __name__ == '__main__':
    problem_table = pd.read_excel('./source/dataset/raw_data/intention/附件一、資料格式_問題類別清單及回覆內容_晧揚回饋版本.xlsx',sheet_name = 0,engine = 'openpyxl')
    for c in ['問題大類類別','問題大類','大類回應方式','大類回應方式.1']:
        problem_table[c] = problem_table[c].fillna(method = 'ffill')
    problem_table = problem_table[['問題大類類別','問題大類','問題編號','問題類別']]
    problem_table = problem_table[problem_table['問題編號'].notnull()]
    problem_table = problem_table.rename(columns = {'問題大類類別':'problem_large_id',\
                                                    '問題大類':'problem_large_name',\
                                                    '問題編號':'predictive_problem_id',
                                                    '問題類別':'predictive_problem_name'})
    problem_table = problem_table[['predictive_problem_id','predictive_problem_name']]

    checkpoint_folder = 'C:\\Users\\cloudy822\\Desktop\\intent classification\\saved\\intent_classification\\2023-08-21_17_51_52\\'
    model_path = os.path.join(checkpoint_folder, 'intent_bert_model')
    token_path = os.path.join(checkpoint_folder, 'intent_bert_model')
    config_path = os.path.join(checkpoint_folder, 'config_for_intention.ini')

    model = BertForSequenceClassifier.from_pretrained(pretrained_model_name_or_path = model_path,num_labels = 15)
    mapping_table = {'SQ001': 0, 'SQ002': 1, 'SQ003': 2, 'SQ004': 3, 'SQ005': 4, 'SQ007': 5, 'SQ008': 6, 'SQ009': 7, 'SQ010': 8, 'SQ011': 9, 'SQ012': 10, 'SQ013': 11, 'SQ014': 12, 'SQ015': 13, 'SQ016': 14}


    input_sentence = ['我的卡片已經卡住','我想要知道客服電話','收費制度如何','收費','費用怎麼算','我發現一張悠遊卡','我的卡片讀取問題,你可以幫我解決嗎?']
    predictive_problem = problem_prediction(input_sentence, model, device = 'cpu', mapping_table = mapping_table)

    test_data = pd.read_excel('./source/dataset/original_data/dialogue_data_20230821.xlsx',engine = 'openpyxl')
    test_data = test_data[test_data['sentence'].str.len()>2]
    input_sentence = test_data['sentence'].tolist()
    predictive_problem = problem_prediction(input_sentence, model, device = 'cuda:0', mapping_table = mapping_table)

    test_data['predictive_problem_id'] = predictive_problem    
    test_data = test_data.merge(problem_table, how = 'left', on = 'predictive_problem_id')
    test_data['isdiff'] = test_data['problem_id'] == test_data['predictive_problem_id']
    test_data.to_excel('./source/ouptut/dialogue_predict_data_20230831.xlsx',engine = 'openpyxl', index = False)
    

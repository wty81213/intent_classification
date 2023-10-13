import time
import re 
import pandas as pd 
import configparser
from tqdm import tqdm
from ast import literal_eval
from intention_dataset import IntentionDataset
from model import BertForSequenceClassifier

import torch
from torch.utils.data import DataLoader

def pre_work_for_intent_recognition(initial_config):
    # setting parameter 
    print('===== setting initial parameter =====')
    PROBLEM_MAPPING_PATH = initial_config['DICTIONARY']['PROBLEM_MAPPING_PATH']
    POSITION_MAPPING_PATH = initial_config['DICTIONARY']['POSITION_MAPPING_PATH']

    MODEL_PATH = initial_config['MODEL']['MODEL_PATH']
    NUM_LABELS = int(initial_config['MODEL']['NUM_LABELS'])
    PREDICTIVE_MAPPING_DICT = literal_eval(initial_config['MODEL']['PREDICTIVE_MAPPING_DICT'])
    
    DEVICE = initial_config['MODEL']['DEVICE']
    device = DEVICE if torch.cuda.is_available() else 'cpu'

    ##### handling predict_mapping_info 
    print('===== handling predict_mapping_info =====')
    problem_mapping_info = pd.read_csv(PROBLEM_MAPPING_PATH)
    position_mapping_info = pd.read_csv(POSITION_MAPPING_PATH)

    problem_mapping_info['value'] = problem_mapping_info['problem_id'] + '_' + problem_mapping_info['problem_name']
    problem_mapping_dict = dict(zip(problem_mapping_info['problem_id'],problem_mapping_info['value']))
    predict_mapping_dict = {v: problem_mapping_dict[k] for k, v in PREDICTIVE_MAPPING_DICT.items()}
    
    ##### handling postion_mapping_info 
    print('===== handling postion_mapping_info  =====')
    position_mapping_info = pd.read_csv(POSITION_MAPPING_PATH)
    position_mapping_info['station_fuzzyname'] = position_mapping_info['station_fuzzyname'].str.split('@')
    position_mapping_info = position_mapping_info.explode('station_fuzzyname')
    
    value_count_ser = position_mapping_info['station_fuzzyname'].value_counts()
    if sum(value_count_ser > 1):
        duplicated_value = value_count_ser[value_count_ser == 1].index.tolist()
        print('Warning: we find some duplicated value {}'.format('|'.join(duplicated_value)))
        position_mapping_info = position_mapping_info.drop_duplicates('station_fuzzyname')
    position_mapping_info['value'] = position_mapping_info['station_id'] + '_' + position_mapping_info['station']
    position_mapping_dict = dict(zip(position_mapping_info['station_fuzzyname'],position_mapping_info['value']))

    #####  load model
    print('===== load model  =====')
    model = BertForSequenceClassifier.from_pretrained(
            pretrained_model_name_or_path = MODEL_PATH,\
            num_labels = NUM_LABELS)

    return {
        'model':model,
        'model_path': MODEL_PATH,
        'device':DEVICE,
        'predict_mapping_dict':predict_mapping_dict,
        'position_mapping_dict':position_mapping_dict
        }

def matching_stations(sentence, position_mapping_dict):
    # setting parameter 
    fuzzyname = list(position_mapping_dict.keys())
    fuzzyname = sorted(fuzzyname, key = lambda x : len(x))[::-1]

    # checking format of sentences
    if isinstance(sentence,str):
        sentence = [sentence]
    else :
        assert isinstance(sentence,list)    

    stations_list = []
    for s in sentence:
        extract_fuzzyname = re.findall('|'.join(fuzzyname), str(s))
        extract_station = [position_mapping_dict[i] for i in extract_fuzzyname]
        extract_station = list(set(extract_station))
        stations_list.append(extract_station)
    
    return stations_list

def problem_prediction(sentence, model, device, model_path, predict_mapping_dict):

    # checking format of sentences
    if isinstance(sentence,str):
        sentence = [sentence]
    else :
        assert isinstance(sentence,list)

    # convert cpu to specific device
    model = model.to(device)
    
    # data preparation
    senti_data  = (sentence,[0]*len(sentence),[0]*len(sentence))
    input_dataset = IntentionDataset(senti_data, token_path = model_path)
    dataloader = DataLoader(input_dataset, batch_size=1, collate_fn=input_dataset.collate_fn)

    # model prediction
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

    # convert idx to problem_id
    total_logits = torch.cat(total_logits, dim = 0).to('cpu')
    pred_labels = torch.argmax(total_logits, dim=1).to('cpu').numpy()
    pred_labels = [predict_mapping_dict[i] for i in pred_labels]

    return pred_labels

def main_for_intent_recognition(sentence, config):
    print('===== predicting problem =====')
    prediction_result = problem_prediction(sentence, 
                                           model = config['model'],
                                           device = config['device'],
                                           model_path = config['model_path'],
                                           predict_mapping_dict = config['predict_mapping_dict'])
    
    print('===== matching_station =====')
    matching_result = matching_stations(sentence,
                                        position_mapping_dict = config['position_mapping_dict'])

    return {
        'predictive_problem':prediction_result,
        'matching_station':matching_result
    }

if __name__ == '__main__':
    start = time.time()
    initial_config = configparser.ConfigParser()
    initial_config.read('./config.ini')
    
    config = pre_work_for_intent_recognition(initial_config)
    
    # sentence = ['前面有人沒結束、畫面無法操作']
    # print(main_for_intent_recognition(sentence, config))

    data = pd.read_excel('./source/附件三、8月驗測資料_0911.xlsx')
    #data = pd.read_excel('C:\\Users\\cloudy822\\Desktop\\intent classification\\source\\dataset\\raw_data\\intention\\dialogue_dataset_20230914.xlsx')
    #data = data.iloc[:10000]
    data = data[['問題編號_調整','問題類別_調整','客戶詢問的問題']]
    sentences = data['客戶詢問的問題'].tolist()
    #sentences = data['sentence'].tolist()
    result = main_for_intent_recognition(sentences, config)
                                         
    data['predict_idx'] = [ i.split('_')[0] for i in result['predictive_problem'] ] 
    data['predict_problem'] = [ i.split('_')[1] for i in result['predictive_problem'] ]                                       
    data['predict_position'] = result['matching_station']
    data['diff'] = (data['問題類別_調整'] ==  data['predict_problem'])
    #data['diff'] = (data['problem_name'] ==  data['predict_problem'])

    data.to_excel('./source/result_0916_F_v1.xlsx', index = False)

    end = time.time()
    print("執行時間：%f 秒" % (end - start))
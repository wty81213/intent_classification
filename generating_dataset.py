import os 
import logging
import pandas as pd

from utils.logger import Logger
from utils.reading_local_file import read_yaml,read_json,create_folder
from data_processing.data_preparation import generating_dataset

args = read_yaml()
Logger(args['env']['mode']).get_logger('root')

if __name__ == "__main__":
    
    logging.info('##### Data Pre-Processing #####')

    dataset_path = args['pre-processing']['dataset_folder']
    if not os.path.exists(dataset_path):
        create_folder(dataset_path)
        data = pd.read_csv('./source/original_data/sentiment_data.csv')
        generating_dataset(data,'text','sentiment_label',[6,2,2],False,args['pre-processing'])

    logging.info('##### setting = {} #####'.format(read_json(dataset_path+'/data_info.json')))        
        
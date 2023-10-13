import os 
os.chdir(r'c:\\Users\\cloudy822\\Desktop\\intent classification')
import logging
from configuration import ConfigParser
from data_loader import IntentionDataLoader
from model import BertForSequenceClassifier
from trainer import Trainer

logger = logging.getLogger(__name__)
if __name__ == '__main__':
    config = ConfigParser('config_for_intention.ini') 

    logger.info('##### loading dataset  #####')
    dataloader = IntentionDataLoader(**config.train_dataloader_config)
    
    logger.info('total size = {}'.format(len(dataloader)*4))
    logger.info('train size = {}'.format(len(dataloader.train_dataloader)*4))
    # logger.info('valid size = {}'.format(len(dataloader.valid_dataloader)*4))
    # logger.info('test size = {}'.format(len(dataloader.test_dataloader)*4))

    valid_dataloader = IntentionDataLoader(**config.valid_dataloader_config)
    logger.info('total size = {}'.format(len(valid_dataloader)*4))
    logger.info('train size = {}'.format(len(valid_dataloader.train_dataloader)*4))

    logger.info('##### loading model    #####')
    model = BertForSequenceClassifier.from_pretrained(**config.model_basic_config)
    logger.debug(model)

    
    logger.info('##### creating trainer #####')

    training_process = Trainer(
        config = config, 
        model = model, 
        train_dataloader = dataloader.train_dataloader, 
        valid_dataloader = valid_dataloader.train_dataloader
    )

    logger.info('##### model training   #####')
    training_process.train()
    
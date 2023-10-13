import pandas as pd 
from .intention_dataset import IntentionDataset
from .sentiment_dataset import SentimentDataset
from .base_dataloader import BaseDataLoader

class SentimentDataLoader(BaseDataLoader):
    def __init__(
        self,
        file_path, 
        token_path, 
        maxlen,
        batch_size, 
        split_weight, 
        num_workers
    ):
    
        self.file_path = file_path
        self.token_path = token_path
        
        senti_table = pd.read_csv(file_path)
        senti_data  = (senti_table['text'].tolist(), senti_table['sentiment_label'].tolist())
        self.dataset = SentimentDataset(senti_data, token_path = token_path, maxlen = maxlen)
        super().__init__(self.dataset , batch_size, split_weight, num_workers,collate_fn = self.dataset.collate_fn)

class IntentionDataLoader(BaseDataLoader):
    def __init__(
        self,
        file_path,
        l1_target_mapping,
        target_mapping,
        token_path, 
        maxlen,
        batch_size, 
        split_weight, 
        num_workers
    ):
        self.file_path = file_path
        self.token_path = token_path

        self.l1_target_mapping = l1_target_mapping
        self.target_mapping = target_mapping 

        intention_table = pd.read_csv(file_path)
        l1_target = [ l1_target_mapping[str(i)] for i in intention_table['problem_large_id'].tolist()]
        target = [ target_mapping[i] for i in intention_table['problem_id'].tolist()]
        intention_data = (intention_table['sentence'].tolist(),\
                          l1_target, target)  
        self.dataset = IntentionDataset(intention_data, token_path = token_path, maxlen = maxlen)
        super().__init__(self.dataset , batch_size, split_weight, num_workers,collate_fn = self.dataset.collate_fn)
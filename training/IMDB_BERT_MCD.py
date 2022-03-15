from src.models.predict import bert_predict

from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense,Dropout, Input
from tensorflow.keras import regularizers

from pathlib import Path

def build():
    
    NUM_SPLITS = 5
    max_len = 512
    num_classes = 2

    from src.datasets.imdb import IMDB

    imdb = IMDB(clean=False)
    X, y, train_index_list, test_index_list, X_eval, y_eval = imdb.getDataSplits(n_splits=NUM_SPLITS, test_size=12500, random_state=1)
    X = np.array(X)
    y = np.array(y)
    
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
            
    test_encodings_tf = tokenizer(X_eval, max_length=max_len,
                            truncation=True,
                            padding='max_length',
                            return_attention_mask=True,
                            return_token_type_ids=False,
                            return_tensors="tf")
    
    from tqdm import tqdm 
    print('train ...')
    for i in tqdm(range(NUM_SPLITS)):
        print(f'#### run Nr. {i}')
        model = TFDistilBertForSequenceClassification.from_pretrained(f'./models/imdb/bert/IMDB_BERT_BL_{i}', num_labels=2)
        
        print('predict')
        df = bert_predict(model, test_encodings_tf, y_eval, T=50)
        
        print('predict ...')
        df.to_pickle(f"pickle/imdb/df_BERT_MCD_{i}.pkl")
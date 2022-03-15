from src.datasets.util import read_files, get_vocab, pad_sequences, text_to_rank, splits, clean_doc, splitsNonInt
from sklearn.model_selection import StratifiedShuffleSplit,train_test_split
from tensorflow import keras
import numpy as np
from src.models.embedding import * 
import yaml
import pandas as pd
from sklearn.utils import shuffle

# load config
with open('config.yaml', 'r') as f:
    conf = yaml.load(f)

DataPath = conf["ToxicDataPath"]

class Toxic:
    
    def __init__(self, clean=True):
        data = pd.read_csv(DataPath + "train.csv")
       
        doc_labels = (data['toxic'] == 1) | (data['severe_toxic']==1) | (data['obscene']==1) | (data['threat']==1) | (data['insult']==1) | (data['identity_hate']==1)
        

        df = pd.DataFrame ({'id': data.id, 'comment_text': data.comment_text, 'label' : doc_labels.map({True: 1, False: 0})})
        df_sort = df.sort_values('label')
        df_sort = df_sort.iloc[-40000: , :]
        
        docs = df_sort['comment_text']
        docs_label = df_sort['label']
        
        docs, docs_label = shuffle(docs, docs_label, random_state=42)
        
        if clean:
            docs_data_clean = [clean_doc(docs) for docs in docs]
        else:    
            docs_data_clean = docs.values.tolist()
            
    
        X_docs, X_test_docs, y_labels, y_test_labels = train_test_split(
            docs_data_clean, docs_label, 
            stratify=docs_label, 
            test_size=10000,
            random_state=42
        )
            
        self.labels = y_labels
        self.data = X_docs
        
        self.test_labels = y_test_labels
        self.test_data = X_test_docs
        
        self.vocab = get_vocab(self.data)
        
    
    def getRankedDataSplits(self, vocab_size, max_sequence_length, n_splits=5, test_size=0.5, random_state=1):
        X = text_to_rank(self.data, self.vocab, vocab_size)
        X = pad_sequences(X, maxlen=max_sequence_length)
        y = keras.utils.to_categorical(self.labels)

        X_eval = text_to_rank(self.test_data, self.vocab, vocab_size)
        X_eval = pad_sequences(X_eval, maxlen=max_sequence_length)
        y_eval = keras.utils.to_categorical(self.test_labels)
        
        X_train, y_train, X_test, y_test = splits(n_splits, X, y, test_size, random_state)
        
        return X_train, y_train, X_test, y_test, X_eval, y_eval
    
    def getRawDataSplits(self, n_splits=5, test_size=0.5, random_state=1):
        X = self.data
        y = keras.utils.to_categorical(self.labels)

        X_eval = self.test_data
        y_eval = keras.utils.to_categorical(self.test_labels)
        
        seq, word_index = get_data(X+X_eval, len(X))
        
        X = seq[:len(X)]
        X_eval = seq[-len(X_eval):]
        
        allData = np.concatenate([X, X_eval])
        X_train, y_train, X_test, y_test = splits(n_splits, X, y, test_size, random_state)
        
        return X_train, y_train, X_test, y_test, X_eval, y_eval, word_index
    
    def getDataSplits(self, n_splits=5, test_size=0.5, random_state=1):
        X = self.data
        y = self.labels

        X_eval = self.test_data
        y_eval = self.test_labels
        
        allData = np.concatenate([X, X_eval])
        train_index_list, test_index_list = splitsNonInt(n_splits, X, y, test_size, random_state)
        
        return X, y, train_index_list, test_index_list, X_eval, y_eval
    
    def getVocab(self):
        return self.vocab
    
    def getData(self):
        return [*self.data, *self.test_data], [*self.labels, *self.test_labels]
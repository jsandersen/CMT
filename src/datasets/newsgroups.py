from src.datasets.util import read_files, get_vocab, pad_sequences, text_to_rank, splits, clean_doc, splitsNonInt
from sklearn.model_selection import StratifiedShuffleSplit,train_test_split
from tensorflow import keras
import numpy as np
from src.models.embedding import * 

from sklearn.datasets import fetch_20newsgroups

class NewsGroups:
    
    def __init__(self, clean=True):
        docs = fetch_20newsgroups(subset='all', shuffle=True, random_state=1) # remove=('headers', 'footers', 'quotes')
        
        if clean:
            docs_data_clean = [clean_doc(doc) for doc in docs.data]
        else:
            docs_data_clean = docs.data
            
        size_train = 9846
        size_val = 4500
        

        X_docs, X_test_docs, y_labels, y_test_labels = train_test_split(
            docs_data_clean, docs.target, 
            stratify=docs.target, 
            test_size=4500,
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
        
    
    
from src.datasets.util import read_files, get_vocab, pad_sequences, text_to_rank, splits, splitsNonInt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import yaml
from tensorflow import keras
from src.models.embedding import * 

# load config
with open('config.yaml', 'r') as f:
    conf = yaml.load(f)
DataPath = conf["IMDBDataPath"]


class IMDB:
    
    def __init__(self, clean=True):
        negative_docs = read_files(DataPath + 'train/neg', clean)
        positive_docs = read_files(DataPath + 'train/pos', clean)
        negative_docs_test = read_files(DataPath + 'test/neg', clean)
        positive_docs_test = read_files(DataPath + 'test/pos', clean)
        
        #print(len(negative_docs), DataPath + 'train/neg')
        
        docs = negative_docs + positive_docs
        docs_labels = [0 for _ in range(len(negative_docs))] + [1 for _ in range(len(positive_docs))] 

        docs_test = negative_docs_test + positive_docs_test
        docks_test_labels = [0 for _ in range(len(negative_docs_test))] + [1 for _ in range(len(positive_docs_test))]
        
        X_docs, X_test_docs, y_labels, y_test_labels = train_test_split(docs_test, docks_test_labels, stratify=docks_test_labels, test_size=0.5, random_state=42)
        
        self.data = docs + X_docs
        self.labels = docs_labels + y_labels
        
        self.test_data = X_test_docs
        self.test_labels = y_test_labels
        
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
        return self.data + self.test_data, self.labels + self.test_labels
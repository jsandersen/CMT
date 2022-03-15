import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

from src.datasets.newsgroups import NewsGroups
from src.datasets.util import splits
from src.models.cnn2 import getCNN2
from src.models.embedding import * 
from src.models.predict import predict_mcdropout
import yaml

import pandas as pd
import tensorflow as tf

from gensim import models

# load config
with open('config.yaml', 'r') as f:
    conf = yaml.load(f)
Word2VecModelPath = conf["Word2VecModelPath"]


# config
RANDOM_STATE = 1

MAX_SEQUENCE_LENGTH = 500

NUM_SPLITS = 5



def build():

    # get data
    print('read data ...')
    
    newsGroups = NewsGroups()
    X_train, y_train, X_test, y_test, X_eval, y_eval, word_index = newsGroups.getRawDataSplits(n_splits=NUM_SPLITS, test_size=4500, random_state=1)
    
    print('create embedding ...')
    # embedding
    w = models.KeyedVectors.load_word2vec_format(Word2VecModelPath, binary=True)
    
    embeddings_index, embedding_dim = get_embeddings_index(w)
    
    w = None

    # load BL model
    print('train model ...')
    
    models_n = []

    for i in range(NUM_SPLITS):
        model = tf.keras.models.load_model(f'models/newsGroups/CNN2_BL_{i}')
        models_n.append(model)
        
    # predict
    print('evaluate ...')
    dfs = []
    for m in range(NUM_SPLITS):
        dfs_parts = []
        s = 500
        j = s
        for i in range(0, 4500, s):
            dfs_n = predict_mcdropout(models_n[m], X_eval[i:j], y_eval[i:j])
            dfs_parts.append(dfs_n)
            print(i, j)
            j+=s
        dfs.append(pd.concat([*dfs_parts], ignore_index=True))

    # save
    print('save as dataframe  ...')
    name = 'CNN2_MCD'
    i = 0
    for df in dfs:
        df.to_pickle(f"pickle/newsGroups/{name}_{i}.pkl")
        i = i+1
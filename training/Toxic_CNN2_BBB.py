import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

from src.datasets.toxic import Toxic
from src.datasets.util import splits
from src.models.cnn2 import getCNN2_BBB
from src.models.embedding import * 
from src.models.predict import predict_bbb
import yaml

import pandas as pd
import tensorflow as tf

from gensim import models

# load conifig
with open('config.yaml', 'r') as f:
    conf = yaml.load(f)
Word2VecModelPath = conf["Word2VecModelPath"]

# config
RANDOM_STATE = 1

MAX_SEQUENCE_LENGTH = 500

NUM_SPLITS = 5
SPLIT_SIZE = 10000

BATCH_SIZE= 100

NUM_EPOCHS = 100

def build():

    
    # get data
    print('read data ...')
    
    toxic = Toxic(clean=True)
    X_train, y_train, X_test, y_test, X_eval, y_eval, word_index = toxic.getRawDataSplits(n_splits=NUM_SPLITS, test_size=SPLIT_SIZE, random_state=RANDOM_STATE)
    
    print('create embedding ...')
    # embedding
    w = models.KeyedVectors.load_word2vec_format(Word2VecModelPath, binary=True)
    
    embeddings_index, embedding_dim = get_embeddings_index(w)
    
    w = None
    
    # training
    print('train model ...')
    
    models_n = []

    for i in range(NUM_SPLITS):
        embedding_layer = get_embedding_layer(word_index, embeddings_index, embedding_dim)
        model = getCNN2_BBB(MAX_SEQUENCE_LENGTH, 2, embedding_layer, 20000)
        model.compile(loss='binary_crossentropy',
                      optimizer=tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95),
                      metrics=['accuracy'])
        history = model.fit(X_train[i], y_train[i],
                  batch_size=BATCH_SIZE,
                  epochs=NUM_EPOCHS,
                  validation_data=(X_test[i], y_test[i])
                 )
        models_n.append(model)

    # predict
    print('evaluate ...')
    dfs = []
    for m in range(NUM_SPLITS):
        dfs_parts = []
        s = 2500
        j = s
        for i in range(0, SPLIT_SIZE, s):
            dfs_n = predict_bbb(models_n[m], X_eval[i:j], y_eval[i:j])
            dfs_parts.append(dfs_n)
            print('#', i, j)
            j+=s
        dfs.append(pd.concat([*dfs_parts], ignore_index=True))

    # save
    print('save as dataframe  ...')
    name = 'CNN2_BBB'
    i = 0
    for df in dfs:
        df.to_pickle(f"pickle/toxic/df_{name}_{i}.pkl")
        i = i+1
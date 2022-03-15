import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

from src.datasets.newsgroups import NewsGroups
from src.datasets.util import splits
from src.models.cnn2 import getCNN2
from src.models.embedding import * 
from src.models.predict import predict_ensamble

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
MAX_EPOCHS=100

NUM_SPLITS = 5

BATCH_SIZE= 100
EMBEDDING_DIM = 100

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
    
    # training
    print('train model ...')
    
    T = 5
    
    models_as_set = []
    for _ in range(NUM_SPLITS):
        embedding_layer = get_embedding_layer(word_index, embeddings_index, embedding_dim)
        models_as = [getCNN2(MAX_SEQUENCE_LENGTH, 20, embedding_layer) for i in range(T)]
        models_as_set.append(models_as)
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    for i in range(NUM_SPLITS):
        for t in range(T):
            models_as_set[i][t].compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95),
                      metrics=['accuracy'])
            models_as_set[i][t].fit(X_train[i], y_train[i],
                  batch_size=BATCH_SIZE,
                  epochs=MAX_EPOCHS,
                  callbacks=[callback],
                  validation_data=(X_test[i], y_test[i])
                 )
            models_as_set[i][t].save(f'pickle/newsGroups/tmp_newsgroups_en_cnn2_{i}_{t}.h5') # save memory
            models_as_set[i][t].save(f'models/newsGroups/model_CNN2_EN_{i}_{t}')
            models_as_set[i][t] = None
            print('###', i, t)

    models_as_set = []
    for i in range(NUM_SPLITS):
        model_as = []
        for t in range(T):
            m = tf.keras.models.load_model(f'pickle/newsGroups/tmp_newsgroups_en_cnn2_{i}_{t}.h5')
            model_as.append(m)
        models_as_set.append(model_as)
            
 
         
    # predict
    print('evaluate ...')
    dfs = []
    for m in range(NUM_SPLITS):
        dfs_parts = []
        s = 500
        j = s
        for i in range(0, 4500, s):
            dfs_n = predict_ensamble(models_as_set[m], X_eval[i:j], y_eval[i:j])
            dfs_parts.append(dfs_n)
            print(i, j)
            j+=s
        dfs.append(pd.concat([*dfs_parts], ignore_index=True))

    # save
    print('save as dataframe  ...')
    name = 'CNN2_EN'

    i = 0
    for df in dfs:
        df.to_pickle(f"pickle/newsGroups/{name}_{i}.pkl")
        i = i+1
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

from src.datasets.newsgroups import NewsGroups
from src.models.cnn1 import getCNN1
from src.models.predict import predict_mcdropout

import tensorflow as tf

def build():
    
    # config
    RANDOM_STATE = 1

    VOCAB_SIZE = 20000
    MAX_SEQUENCE_LENGTH = 500
    NUM_SPLITS = 5



    # get data
    newsGroups = NewsGroups()

    X_train_set, y_train_set, X_test_set, y_test_set, X_val, y_val = newsGroups.getRankedDataSplits(
        vocab_size=VOCAB_SIZE,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        n_splits=NUM_SPLITS,
        test_size=4500,
        random_state=RANDOM_STATE
    )

    # training
    models_n = []

    for i in range(NUM_SPLITS):
        model = tf.keras.models.load_model(f'models/newsGroups/CNN1_BL_{i}')
        models_n.append(model)

    # predict 
    dfs = [predict_mcdropout(models_n[i], X_val, y_val) for i in range(NUM_SPLITS)]

    #save df
    name = 'CNN1_MCD'

    i = 0
    for df in dfs:
        df.to_pickle(f"pickle/newsGroups/{name}_{i}.pkl")
        i = i+1
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

from src.datasets.imdb import IMDB
from src.models.cnn1 import getCNN1
from src.models.predict import predict_mcdropout
import tensorflow as tf

def build():
    
    # config
    RANDOM_STATE = 1

    VOCAB_SIZE = 20000
    MAX_SEQUENCE_LENGTH = 500

    NUM_SPLITS = 5
    SPLIT_SIZE = 12500

    BATCH_SIZE= 100
    EMBEDDING_DIM = 100
    NUM_EPOCHS = 100

    # get data
    imdb = IMDB()

    X_train, y_train, X_test, y_test, X_val, y_val = imdb.getRankedDataSplits(
        vocab_size=VOCAB_SIZE,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        n_splits=NUM_SPLITS,
        test_size=SPLIT_SIZE,
        random_state=RANDOM_STATE
    )

    # load BL model
    models_n = []
    
    for i in range(NUM_SPLITS):
        model = tf.keras.models.load_model(f'models/imdb/CNN1_BL_{i}')
        models_n.append(model)

    # predict 
    dfs = [predict_mcdropout(models_n[i], X_val, y_val) for i in range(NUM_SPLITS)]

    # save
    name = 'CNN1_MCD'
    i = 0
    for df in dfs:
        df.to_pickle(f"pickle/imdb/{name}_{i}.pkl")
        i = i+1
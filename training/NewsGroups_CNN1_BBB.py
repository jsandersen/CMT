import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

from src.datasets.newsgroups import NewsGroups
from src.models.cnn1 import getCNN1_BBB
from src.models.predict import predict_bbb

import tensorflow as tf

def build():
    
    RANDOM_STATE = 1

    VOCAB_SIZE = 20000
    MAX_SEQUENCE_LENGTH = 500

    NUM_SPLITS = 5

    BATCH_SIZE= 100
    EMBEDDING_DIM = 100
    MAX_EPOCHS = 100

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
    historys_n = []

    for i in range(NUM_SPLITS):
        model = getCNN1_BBB(vocab_size=VOCAB_SIZE, embedding_dims=EMBEDDING_DIM, max_sequence_len=MAX_SEQUENCE_LENGTH, n_classes=20, size_taining_set=9846)
        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(0.0001),
                      metrics=['accuracy'])
        history = model.fit(X_train_set[i], y_train_set[i],
                  batch_size=BATCH_SIZE,
                  epochs=NUM_EPOCHS,
                  validation_data=(X_test_set[i], y_test_set[i]),
                 )

        models_n.append(model)
        historys_n.append(history)

    # predict 
    dfs = [predict_bbb(models_n[i], X_val, y_val) for i in range(NUM_SPLITS)]

    #save df
    name = 'CNN1_BBB'
    i = 0
    for df in dfs:
        df.to_pickle(f"pickle/newsGroups/{name}_{i}.pkl")
        i = i+1
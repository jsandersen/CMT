import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

from src.datasets.newsgroups import NewsGroups
from src.models.cnn1 import getCNN1
from src.models.predict import predict_ensamble

import tensorflow as tf

def build():
    
    # config
    RANDOM_STATE = 1

    VOCAB_SIZE = 20000
    MAX_SEQUENCE_LENGTH = 500

    NUM_SPLITS = 5

    BATCH_SIZE= 100
    EMBEDDING_DIM = 100
    NUM_EPOCHS = 100

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
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    
    T = 5
    models_as_set = []
    for _ in range(NUM_SPLITS):
        models_as = [getCNN1(vocab_size=VOCAB_SIZE, embedding_dims=EMBEDDING_DIM, max_sequence_len=MAX_SEQUENCE_LENGTH, n_classes=20) for i in range(T)]
        models_as_set.append(models_as)
    
    for i in range(NUM_SPLITS):
        for t in range(T):
            models_as_set[i][t].compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(0.0001),
                      metrics=['accuracy'])
            models_as_set[i][t].fit(X_train_set[i], y_train_set[i],
                  batch_size=BATCH_SIZE,
                  epochs=NUM_EPOCHS,
                  callbacks=[callback],
                  validation_data=(X_test_set[i], y_test_set[i])
                 )
            models_as_set[i][t].save(f'models/newsGroups/model_CNN1_EN_{i}_{t}')
            print('###', i, t)
    

    # predict 
    dfs = []
    for i in range(len(models_as_set)):
        dfs.append(predict_ensamble(models_as_set[i], X_val, y_val))

    #save df
    name = 'CNN1_EN'

    i = 0
    for df in dfs:
        df.to_pickle(f"pickle/newsGroups/{name}_{i}.pkl")
        i = i+1
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()


from src.datasets.toxic import Toxic
from src.models.cnn1 import getCNN1
from src.models.predict import predict_ensamble
import tensorflow as tf
import pandas as pd

def build():
    
    # config
    RANDOM_STATE = 1

    VOCAB_SIZE = 20000
    MAX_SEQUENCE_LENGTH = 500

    NUM_SPLITS = 5
    SPLIT_SIZE = 10000

    BATCH_SIZE= 100
    EMBEDDING_DIM = 100
    NUM_EPOCHS = 100

    # get data
    toxic = Toxic(clean=True)

    X_train, y_train, X_test, y_test, X_val, y_val = toxic.getRankedDataSplits(
        vocab_size=VOCAB_SIZE,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        n_splits=NUM_SPLITS,
        test_size=SPLIT_SIZE,
        random_state=RANDOM_STATE
    )

    # get models
    T = 5
    models_as_set = []
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    
    for _ in range(NUM_SPLITS):
        models_as = [getCNN1(
                vocab_size=VOCAB_SIZE, 
                embedding_dims=EMBEDDING_DIM, 
                max_sequence_len=MAX_SEQUENCE_LENGTH,
                n_classes=2,
            ) for i in range(T)]
        models_as_set.append(models_as)
    
    # train
    for i in range(NUM_SPLITS):
        for t in range(T):
            models_as_set[i][t].compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(0.0001),
                  metrics=['acc'])
            models_as_set[i][t].fit(X_train[i], y_train[i], validation_data=(X_test[i], y_test[i]), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, callbacks=[callback],)
            
            models_as_set[i][t].save(f'models/toxic/model_CNN1_EN_{i}_{t}')
            print('#', i, t)
        
    # predict 
    dfs = []
    for m in range(NUM_SPLITS):
        dfs_parts = []
        s = 2500
        j = s
        for i in range(0, SPLIT_SIZE, s):
            dfs_n = predict_ensamble(models_as_set[m], X_val[i:j], y_val[i:j])
            dfs_parts.append(dfs_n)
            j+=s
        dfs.append(pd.concat([*dfs_parts], ignore_index=True))

    # save
    name = 'CNN1_EN'
    i = 0
    for df in dfs:
        df.to_pickle(f"pickle/toxic/df_{name}_{i}.pkl")
        i = i+1
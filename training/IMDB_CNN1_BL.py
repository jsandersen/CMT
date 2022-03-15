import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

from src.datasets.imdb import IMDB
from src.models.cnn1 import getCNN1
from src.models.predict import predict


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
    print('load data ...')
    imdb = IMDB()

    X_train, y_train, X_test, y_test, X_val, y_val = imdb.getRankedDataSplits(
        vocab_size=VOCAB_SIZE,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        n_splits=NUM_SPLITS,
        test_size=SPLIT_SIZE,
        random_state=RANDOM_STATE
    )

    # training
    models_n = []
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    
    print('train ...')
    for i in range(NUM_SPLITS):
        model = getCNN1(
            vocab_size=VOCAB_SIZE, 
            embedding_dims=EMBEDDING_DIM, 
            max_sequence_len=MAX_SEQUENCE_LENGTH,
            n_classes=2
        )
        
        model.compile(loss='binary_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(0.0001),
                      metrics=['accuracy'])
        history = model.fit(X_train[i], y_train[i],
                  batch_size=BATCH_SIZE,
                  epochs=NUM_EPOCHS,
                  validation_data=(X_test[i], y_test[i]),
                  callbacks=[callback],
                 )
       
        models_n.append(model)
        model.save(f'models/imdb/CNN1_BL_{i}')

    
    # predict 
    print('predict ...')
    dfs = [predict(models_n[i], X_val, y_val) for i in range(1)]
    
    # save
    print('save predict ...')
    name = 'CNN1_BL'
    i = 0
    for df in dfs:
        df.to_pickle(f"pickle/imdb/{name}_{i}.pkl")
        i = i+1
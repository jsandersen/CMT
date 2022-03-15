import tensorflow as tf
tf.compat.v1.disable_v2_behavior()


from src.datasets.toxic import Toxic
from src.models.cnn1 import getCNN1_BBB
from src.models.predict import predict_bbb

import tensorflow as tf

def build():
    
    # config
    RANDOM_STATE = 1

    VOCAB_SIZE = 20000
    MAX_SEQUENCE_LENGTH = 500

    NUM_SPLITS = 5
    SPLIT_SIZE = 10000

    BATCH_SIZE= 100
    EMBEDDING_DIM = 100
    NUM_EPOCHS = 10

    # get data
    print('load data ...')
    
    toxic = Toxic(clean=True)
    
    X_train, y_train, X_test, y_test, X_val, y_val = toxic.getRankedDataSplits(
        vocab_size=VOCAB_SIZE,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        n_splits=NUM_SPLITS,
        test_size=SPLIT_SIZE,
        random_state=RANDOM_STATE
    )

    # training
    models_n = []
    historys_n = [] 
    
    print('train ...')
    for i in range(NUM_SPLITS):
        model = getCNN1_BBB(
            vocab_size=VOCAB_SIZE,
            embedding_dims=EMBEDDING_DIM, 
            max_sequence_len=MAX_SEQUENCE_LENGTH,
            n_classes=2,
            size_taining_set=20000
        )
        
        model.compile(loss='binary_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(0.0001),
                      metrics=['accuracy'])
        history = model.fit(X_train[i], y_train[i],
                  batch_size=BATCH_SIZE,
                  epochs=NUM_EPOCHS,
                  validation_data=(X_test[i], y_test[i]),
                 )
        models_n.append(model)
        historys_n.append(history)
        
        model.save(f'models/toxic/model_CNN1_BBB_{i}')

    # predict 
    print('predict ...')
    dfs = [predict_bbb(models_n[i], X_val, y_val) for i in range(NUM_SPLITS)]
    
    # save
    print('save predict ...')
    name = 'CNN1_BBB'
    i = 0
    for df in dfs:
        df.to_pickle(f"pickle/toxic/df_{name}_{i}.pkl")
        i = i+1
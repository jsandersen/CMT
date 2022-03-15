from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential

import tensorflow as tf
import tensorflow_probability as tfp

l2_reg = 0.00001
nb_feature_maps = 128

def getCNN1(vocab_size, embedding_dims, max_sequence_len, n_classes):
    model = Sequential()
    model.add(Embedding(vocab_size,
                        embedding_dims,
                        input_length=max_sequence_len,
                       ))
    model.add(Dropout(0.4))
    model.add(Conv1D(nb_feature_maps,
                     3,
                     padding='valid',
                     activation='relu',
                     kernel_regularizer=l2(l2_reg)))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(0.4))
    model.add(Dense(n_classes, activation='softmax'))
    return model

def getCNN1_BBB(vocab_size, embedding_dims, max_sequence_len, n_classes, size_taining_set):
    def kernel_divergence_fn(q, p, _):
        return tfp.distributions.kl_divergence(q, p) / tf.cast(size_taining_set, tf.float32)
    
    model = Sequential()
    model.add(Embedding(vocab_size,
                        embedding_dims,
                        input_length=max_sequence_len,
                       ))
    model.add(tfp.layers.Convolution1DFlipout(nb_feature_maps,
                     3,
                     padding='valid',
                     activation='relu',
                     kernel_divergence_fn=kernel_divergence_fn))
    model.add(GlobalMaxPooling1D())
    model.add(tfp.layers.DenseFlipout(128, activation='relu', kernel_divergence_fn=kernel_divergence_fn))
    model.add(tfp.layers.DenseFlipout(64, activation='relu', kernel_divergence_fn=kernel_divergence_fn))
    model.add(tfp.layers.DenseFlipout(n_classes, activation='softmax', kernel_divergence_fn=kernel_divergence_fn))
    return model
    
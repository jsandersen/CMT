from tensorflow.keras.layers import Dropout, Concatenate, Conv1D, Embedding, concatenate, Input, Flatten, MaxPooling1D, Dense, GlobalMaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.constraints import max_norm

import tensorflow as tf
import tensorflow_probability as tfp

activation='relu'
nb_feature_maps = 100

def getCNN2(max_sequence_len, n_classes, embedding_layer):
    main_input = Input(shape=(max_sequence_len,), dtype='int32')
    e = embedding_layer(main_input)
    
    x1 = Dropout(0.5)(e)
    x1 = Conv1D(nb_feature_maps, 3, padding='valid', activation='relu', strides=1, kernel_constraint=max_norm(3))(x1)
    x1 = MaxPooling1D(max_sequence_len - 3 + 1)(x1)
    x1 = Flatten()(x1)

    x2 = Dropout(0.5)(e)
    x2 = Conv1D(nb_feature_maps, 4, padding='valid', activation='relu', strides=1, kernel_constraint=max_norm(3))(x2)
    x2 = MaxPooling1D(max_sequence_len - 4 + 1)(x2)
    x2 = Flatten()(x2)

    x3 = Dropout(0.5)(e)
    x3 = Conv1D(nb_feature_maps, 5, padding='valid', activation='relu', strides=1, kernel_constraint=max_norm(3))(x3)
    x3 = MaxPooling1D(max_sequence_len - 5 + 1)(x3)
    x3 = Flatten()(x3)

    x = Concatenate()([x1, x2, x3])
    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=[main_input], outputs=[x])

    return model


def getCNN2_BBB(max_sequence_len, n_classes, embedding_layer, size_taining_set):
    def kernel_divergence_fn(q, p, _):
        return tfp.distributions.kl_divergence(q, p) / tf.cast(size_taining_set, tf.float32)
    
    main_input = Input(shape=(max_sequence_len,), dtype='int32')
    e = embedding_layer(main_input)
    
    x1 = Dropout(0.5)(e)
    x1 = tfp.layers.Convolution1DFlipout(nb_feature_maps, 3, padding='valid', activation='relu', strides=1, kernel_divergence_fn=kernel_divergence_fn)(x1)
    x1 = MaxPooling1D(max_sequence_len - 3 + 1)(x1)
    x1 = Flatten()(x1)

    x2 = Dropout(0.5)(e)
    x2 = tfp.layers.Convolution1DFlipout(nb_feature_maps, 4, padding='valid', activation='relu', strides=1, kernel_divergence_fn=kernel_divergence_fn)(x2)
    x2 = MaxPooling1D(max_sequence_len - 4 + 1)(x2)
    x2 = Flatten()(x2)

    x3 = Dropout(0.5)(e)
    x3 = tfp.layers.Convolution1DFlipout(nb_feature_maps, 5, padding='valid', activation='relu', strides=1, kernel_divergence_fn=kernel_divergence_fn)(x3)
    x3 = MaxPooling1D(max_sequence_len - 5 + 1)(x3)
    x3 = Flatten()(x3)

    x = Concatenate()([x1, x2, x3])
    x = tfp.layers.DenseFlipout(n_classes, activation='softmax', kernel_divergence_fn=kernel_divergence_fn)(x)

    model = Model(inputs=[main_input], outputs=[x])

    return model
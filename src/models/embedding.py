import tensorflow as tf
import numpy as np

MAX_NUM_WORDS = 30000
MAX_SEQUENCE_LENGTH = 500

def get_data(texts, training_size):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts[:training_size])
    
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    return tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH), word_index

def get_embeddings_index(model):
    embeddings_index = model.wv.vocab
    
    for word, vocab in embeddings_index.items():
        embeddings_index[word] = model.wv.vectors[vocab.index]
    return embeddings_index, model.vector_size

def get_embedding_layer(word_index, embedding_index, embedding_dim, static=False):
    num_words = min(MAX_NUM_WORDS, len(word_index))
    embedding_matrix = np.zeros((num_words+1, embedding_dim))
    for word, i in word_index.items():
        if i > MAX_NUM_WORDS:
            continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return tf.keras.layers.Embedding(num_words+1,
        embedding_dim,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=static)
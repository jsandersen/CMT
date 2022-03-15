import os
import re
import collections
from tensorflow import keras
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def read_files(path, clean):
    documents = list()
    if os.path.isdir(path):
        for filename in os.listdir(path):
            with open('%s/%s' % (path, filename), encoding="utf8") as f:
                doc = f.read()
                if clean:
                    doc = clean_doc(doc)
                documents.append(doc)
    
    if os.path.isfile(path):        
        with open(path, encoding='iso-8859-1') as f:
            doc = f.readlines()
            for line in doc:
                documents.append(clean_doc(line))
    return documents

stop_words = ['i','me','my','myself','we','our','ours','ourselves','you','your','yours',
                  'yourself','yourselves','he','him','his','himself','she','her','hers','herself',
                  'it','its','itself','they','them','their','theirs','themselves','what','which',
                  'who','whom','this','that','these','those','am','is','are','was','were','be',
                  'been','being','have','has','had','having','do','does','did','doing','a','an',
                  'the','and','but','if','or','because','as','until','while','of','at','by','for',
                  'with','about','against','between','into','through','during','before','after',
                  'above','below','to','from','up','down','in','out','on','off','over','under',
                  'again','further','then','once','here','there','when','where','why','how','all',
                  'any','both','each','few','more','most','other','some','such','no','nor','not',
                  'only','own','same','so','than','too','very','s','t','can','will','just','don',
                  'should','now','d','ll','m','o','re','ve','y','ain','aren','couldn','didn',
                  'doesn','hadn','hasn','haven','isn','ma','mightn','mustn','needn','shan',
                  'shouldn','wasn','weren','won','wouldn']

def clean_doc(doc):
    doc = doc.lower()
    doc = re.sub(r"´,`", "\'", doc)
    doc = re.sub(r"\'s", " is", doc)
    doc = re.sub(r"\'ve", " have", doc)
    doc = re.sub(r"n\'t", " not", doc)
    doc = re.sub(r"\'re", " are", doc)
    doc = re.sub(r"\'d", " \'d", doc)
    doc = re.sub(r"\'ll", " will", doc)
    doc = re.sub(r",", " ", doc)
    doc = re.sub(r"!", " ! ", doc)
    doc = re.sub(r"\(", "", doc)
    doc = re.sub(r"\)", "", doc)
    doc = re.sub(r"\?", r" \? ", doc)
    doc = re.sub(r"\s{2,}", " ", doc)
    doc = re.sub(r"\?", " ", doc)
    doc = re.sub(r"[^A-Za-z0-9(),!?\\`]", " ", doc)
    
    doc = re.sub(r" br ", "", doc)
    
    tokens = doc.split()
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def get_vocab(dataset):
    vocab = {}

    for example in dataset:
        example_as_list = example.split()
        for word in example_as_list:
            vocab[word] = 0

    for example in dataset:
        example_as_list = example.split()
        for word in example_as_list:
            vocab[word] += 1
    
    return collections.OrderedDict(sorted(vocab.items(), key=lambda x: x[1], reverse=True))

def text_to_rank(dataset, _vocab, desired_vocab_size=5000):
    _dataset = dataset[:]
    vocab_ordered = list(_vocab)
    count_cutoff = _vocab[vocab_ordered[desired_vocab_size-1]]
    
    word_to_rank = {}
    for i in range(len(vocab_ordered)):
        word_to_rank[vocab_ordered[i]] = i + 1
    
    for i in range(50):
        _vocab[vocab_ordered[desired_vocab_size+i]] -= 0.1
    
    for i in range(len(_dataset)):
        example = _dataset[i]
        example_as_list = example.split()
        for j in range(len(example_as_list)):
            try:
                if _vocab[example_as_list[j]] >= count_cutoff:
                    example_as_list[j] = word_to_rank[example_as_list[j]] 
                else:
                    example_as_list[j] = desired_vocab_size
            except:
                example_as_list[j] = desired_vocab_size
        _dataset[i] = example_as_list

    return _dataset

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

def splits(n_splits, X, y, test_size, random_state):
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    X_train_set = []
    X_test_set = []

    y_train_set = []
    y_test_set = []

    for train_index, test_index in sss.split(X, y):
        X_train_tmp, X_test_tep = X[train_index], X[test_index]
        y_train_tmp, y_test_tmp = y[train_index], y[test_index]

        X_train_set.append(X_train_tmp)
        X_test_set.append(X_test_tep)
        y_train_set.append(y_train_tmp)
        y_test_set.append(y_test_tmp)

    return X_train_set, y_train_set, X_test_set, y_test_set


def splitsNonInt(n_splits, X, y, test_size, random_state):
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    train_index_list = [] 
    test_index_list = []

    for train_index, test_index in sss.split(X, y):
        train_index_list.append(train_index)
        test_index_list.append(test_index)

    return train_index_list, test_index_list
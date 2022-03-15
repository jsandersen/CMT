from src.models.predict import bert_predict

from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense,Dropout, Input
from tensorflow.keras import regularizers

import tensorflow as tf

def build():
    
    NUM_SPLITS = 5
    max_len = 512
    num_classes = 2

    from src.datasets.imdb import IMDB

    imdb = IMDB(clean=False)
    X, y, train_index_list, test_index_list, X_eval, y_eval = imdb.getDataSplits(n_splits=NUM_SPLITS, test_size=12500, random_state=1)
    X = np.array(X)
    y = np.array(y)
    
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
            
    test_encodings_tf = tokenizer(X_eval, max_length=max_len,
                            truncation=True,
                            padding='max_length',
                            return_attention_mask=True,
                            return_token_type_ids=False,
                            return_tensors="tf")

    # training
    
    for i in range(NUM_SPLITS):
                    
        train_texts, train_labels= X[train_index_list[i]], y[train_index_list[i]]
        test_texts, test_labels = X[test_index_list[i]], y[test_index_list[i]]
        
        train_encodings = tokenizer(train_texts.tolist(), max_length=max_len,
                            truncation=True,
                            padding='max_length',
                            return_attention_mask=True,
                            return_token_type_ids=False)
        val_encodings = tokenizer(test_texts.tolist(), max_length=max_len,
                            truncation=True,
                            padding='max_length',
                            return_attention_mask=True,
                            return_token_type_ids=False)
        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            train_labels
        ))
        val_dataset = tf.data.Dataset.from_tensor_slices((
            dict(val_encodings),
            test_labels
        ))
            
        for t in range(NUM_SPLITS):
            print('###', i, t)

            from transformers import TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments


            training_args = TFTrainingArguments(
                output_dir=f'./models/imdb/finetuned/Imdb_BERT_EN__{i}_{t}',          # output directory
                num_train_epochs=3,              # total number of training epochs
                per_device_train_batch_size=16,  # batch size per device during training
                per_device_eval_batch_size=64,   # batch size for evaluation
                warmup_steps=500,                # number of warmup steps for learning rate scheduler
                weight_decay=0.01,               # strength of weight decay
                logging_dir=f'./models/imdb/finetuned/Imdb_BERT_EN_{i}_{t}',            # directory for storing logs
                logging_steps=2000,
                save_total_limit = 1
            )

            with training_args.strategy.scope():
                model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

            trainer = TFTrainer(
                model=model,                         # the instantiated 🤗 Transformers model to be trained
                args=training_args,                  # training arguments, defined above
                train_dataset=train_dataset,         # training dataset
                eval_dataset=val_dataset             # evaluation dataset
            )
            
            print('pretrain ...')
            trainer.train()
            
            model.save_pretrained(f'./models/imdb/bert/Imdb_BERT_EN_{i}_{t}')
            
            del model
            del trainer
            del training_args


    # predict 
    dfs = []
    for i in range(len(models_as_set)):
        dfs.append(predict_ensamble(models_as_set[i], X_val, y_val))

    #save df
    name = 'BERT_EN'

    i = 0
    for df in dfs:
        df.to_pickle(f"pickle/imdb/{name}_{i}.pkl")
        i = i+1
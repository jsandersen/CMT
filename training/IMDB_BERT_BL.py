from src.models.predict import bert_predict

from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense,Dropout, Input
from tensorflow.keras import regularizers

from pathlib import Path

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
    
    from tqdm import tqdm 
    print('train ...')
    for i in tqdm(range(NUM_SPLITS)):
        print(f'#### run Nr. {i}')
    
        train_texts, train_labels= X[train_index_list[i]], y[train_index_list[i]]
        test_texts, test_labels = X[test_index_list[i]], y[test_index_list[i]]
        
        print('encode ...')
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


        import tensorflow as tf

        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            train_labels
        ))
        val_dataset = tf.data.Dataset.from_tensor_slices((
            dict(val_encodings),
            test_labels
        ))

        from transformers import TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments
        
        
        training_args = TFTrainingArguments(
            output_dir=f'./models/imdb/finetuned/IMDB_BERT_BL_{i}',          # output directory
            num_train_epochs=3,              # total number of training epochs
            per_device_train_batch_size=16,  # batch size per device during training
            per_device_eval_batch_size=64,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir=f'./models/imdb/finetuned/IMDB_BERT_BL_{i}',            # directory for storing logs
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
        
        model.save_pretrained(f'./models/imdb/bert/IMDB_BERT_BL_{i}', num_labels=2)
        
        del trainer
        del train_dataset
        del val_dataset
        
        print('predict')
        df = bert_predict(model, test_encodings_tf, y_eval)
        
        print('to_pickle ...')
        df.to_pickle(f"pickle/imdb/BERT_BL_{i}.pkl")
   
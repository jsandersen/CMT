
# Efficient, Uncertainty-based Moderation of Neural Networks Text Classifiers
Requires Python 3.8 and Jupyter Notebooks
 
## Content

File/Folder|  Description
--- | --- 
`./evaluation` | Scripts to run the evaluation (Jupyter Notebooks). Results of the pure automated models can be found  in the `[Dataset]_[UncertaintyTechnique]_[Model].ipynb` files, the saturation points and accuracies in `[Dataset]_[Model].ipynb`.
`./src` | Source code of the models / pre-processing steps / uncertainty measurements 
`./training` | Scripts to run model training
`./pickle` |  Dataframes containing the predictions and uncertainty estimates of the held-out evaluation datasets
`./config.yaml` | Configuration file for specifying the path of the IMDB/HateSpeech data records and Word2Vec the model
`./run_train.py` | Entry point (Main): execute to re-run one or multiple experiments 



## Required External Files 
- IMDB Dataset: http://ai.stanford.edu/~amaas/data/sentiment/
- Hate Speech Dataset : https://github.com/tianqwang/Toxic-Comment-Classification-Challenge/tree/master/data
- Word2Vec Model: https://github.com/mmihaltz/word2vec-GoogleNews-vectors

The root folder of the dataset files and the Word2Vec model have to be configured in the `.config.yaml` file. 

## Run Experiments

Execute `python run_train.py` to re-run single or multiple experiments. The file looks as follows:

```
from training.Toxic_BERT_BL import build
build()
```

In order to re-run a specific experiment, the import statement has to be edited according to the following scheme: 
`from training.{IMDB|NewsGroups|Toxic}_{CNN1|CNN2|BERT}_{BL|BBB|MCD|EN} import build`

For example, to run the experiment using the Baseline, the IMDB dataset, and CNN use:

```
from training.IMDB_CNN1_BL import build
build()
```

The predictions of an evaluation run  are saved  as dataframes in `./pickle`. When completed the corresponding evaluation script can be executed (Jupyter Notebooks).  

**Note**: In order to run MCD, the corresponding BL must first be run. 

## System Specs
 
GPU: Intel® Xeon® Gold 5115 Processor 10-core 2.40GHz 13.75MB Cache (85W)

RAM: 2666MHz DDR4 ECC Registered DIMM


## Model Description

### CNN (CNN1)
````
Model: "CNN"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 400, 100)          500000    
_________________________________________________________________
dropout (Dropout)            (None, 400, 100)          0         
_________________________________________________________________
conv1d (Conv1D)              (None, 398, 128)          38528     
_________________________________________________________________
global_max_pooling1d (Global (None, 128)               0         
_________________________________________________________________
dense (Dense)                (None, 128)               16512     
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 64)                8256      
_________________________________________________________________
dropout_2 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 130       
=================================================================
Total params: 563,426
Trainable params: 563,426
Non-trainable params: 0
_________________________________________________________________
````


### KimCNN (CNN2)

````
`Model: "KimCNN"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 500)]        0                                            
__________________________________________________________________________________________________
embedding_35 (Embedding)        (None, 500, 300)     9000300     input_1[0][0]                    
__________________________________________________________________________________________________
dropout_105 (Dropout)           (None, 500, 300)     0           embedding_35[0][0]               
__________________________________________________________________________________________________
dropout_106 (Dropout)           (None, 500, 300)     0           embedding_35[0][0]               
__________________________________________________________________________________________________
dropout_107 (Dropout)           (None, 500, 300)     0           embedding_35[0][0]               
__________________________________________________________________________________________________
conv1d_35 (Conv1D)              (None, 498, 100)     90100       dropout_105[0][0]                
__________________________________________________________________________________________________
conv1d_36 (Conv1D)              (None, 497, 100)     120100      dropout_106[0][0]                
__________________________________________________________________________________________________
conv1d_37 (Conv1D)              (None, 496, 100)     150100      dropout_107[0][0]                
__________________________________________________________________________________________________
max_pooling1d (MaxPooling1D)    (None, 1, 100)       0           conv1d_35[0][0]                  
__________________________________________________________________________________________________
max_pooling1d_1 (MaxPooling1D)  (None, 1, 100)       0           conv1d_36[0][0]                  
__________________________________________________________________________________________________
max_pooling1d_2 (MaxPooling1D)  (None, 1, 100)       0           conv1d_37[0][0]                  
__________________________________________________________________________________________________
flatten (Flatten)               (None, 100)          0           max_pooling1d[0][0]              
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 100)          0           max_pooling1d_1[0][0]            
__________________________________________________________________________________________________
flatten_2 (Flatten)             (None, 100)          0           max_pooling1d_2[0][0]            
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 300)          0           flatten[0][0]                    
                                                                 flatten_1[0][0]                  
                                                                 flatten_2[0][0]                  
__________________________________________________________________________________________________
dropout_108 (Dropout)           (None, 300)          0           concatenate[0][0]                
__________________________________________________________________________________________________
dense_105 (Dense)               (None, 2)            602         dropout_108[0][0]                
==================================================================================================
Total params: 9,361,202
Trainable params: 360,902
Non-trainable params: 9,000,300
````

### DistilBert

````
Model: "DistilBert"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
distilbert (TFDistilBertMain multiple                  66362880  
_________________________________________________________________
pre_classifier (Dense)       multiple                  590592    
_________________________________________________________________
classifier (Dense)           multiple                  1538      
_________________________________________________________________
dropout_19 (Dropout)         multiple                  0         
=================================================================
Total params: 66,955,010
Trainable params: 66,955,010
Non-trainable params: 0
````

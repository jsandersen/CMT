import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import warnings
from scipy import stats
from scipy.stats import entropy

T = 50

class Predictor:
    f = None
    
    def __init__(self, model):
        #self.f = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])
        self.model = model
        
    def predict_with_uncertainty(self, x, n_iter, dropout):
        predictions = []
        for i in range(0, n_iter):
            predictions.append(self.f([x, dropout]))        
        return np.array(predictions)

    def predict(self, x):
        #predictions = self.f([x, False])
        predictions = [self.model.predict(x)]
        return np.array(predictions)

class EnsamblePredictor:
    f_assemble = None
    
    def __init__(self, models):
        self.f_assemble = [K.function([m.layers[0].input, K.learning_phase()], [m.layers[-1].output]) for m in models]
    
    def predict(self, x):
        predictions = []
        for f in self.f_assemble:
            predictions.append(f([x, False]))        
        return np.array(predictions)
    
## predict

def _addPredictions(df, mean_prob, y_val, onehot=True):
    if not onehot:
        y_val = to_categorical(y_val)
    
    df['y_hat'] = y_val.argmax(axis=1).flatten()
    if len(y_val[0]) == 2:
        df['p_0'] = mean_prob[:,:, :1].flatten()
        df['p_1'] = 1 - df['p_0']
    df['p'] = mean_prob.max(axis=2).flatten()
    df['y'] = mean_prob.argmax(axis=2).flatten()
    df['defect'] = df['y_hat'] != df['y']
    return df

def _addSimpleScores(df, prob, mean_prob, shape):
    # Defect Eval.
    df['c_false'] =  np.where(df['defect'], 1, 0)
    df['p_defect'] =  np.where(df['defect'], -1*(1-df['p']),  -1*df['p'] )
    df['c_true'] =  np.where(df['defect'], 0, 1)
    df['p_correct'] =  np.where(df['defect'] == False, df['p'],  1-df['p'] )

    # Least Coeficient
    df['u_lc'] = 1 - df['p']
    
    # Highest Margin
    u_diff = []
    for i in range(shape):
        p = mean_prob[:, i][0]
        s = sorted(list(p))[-2:]
        u = 1 - (s[-1] - s[0])
        u_diff.append(u)
    df['u_hm'] = u_diff
    
    # Entropy
    df['u_e'] = entropy(mean_prob.T, base=2).flatten()
    
    return df

def _addAdvancedScores(df, prob, mean_prob, var_prob):
    # Variance
    df['u_var'] = var_prob.mean(axis=2).flatten()
    
    # Mutual Information (BALD)
    def f(x) :
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return  x * np.where(x != 0, np.log2(x), 0)
        
    bald = np.apply_along_axis(f, 0, prob).sum(axis=0).sum(axis=2) / prob.shape[0]
    df['u_bald'] = (df['u_e'] + bald[0])
    
    # Variation Ratio
    vr = []
    for i in range(prob.shape[2]): #N
        vr_i = []
        for j in range(prob.shape[0]):
            p = prob[j, 0, i]
            arg = p.argmax(axis=0)
            vr_i.append(arg)
        vr.append(1 - (vr_i.count(stats.mode(vr_i)[0][0]) / len(vr_i)))
    df['u_vr'] = vr
    
    # Uncertainty Kwon
    epis = np.mean(prob**2, axis=0) - np.mean(prob, axis=0)**2
    df['u_ea'] = np.mean(epis + np.mean(prob*(1-prob), axis=0), axis=2).reshape(prob.shape[2])
    
    def f(x):
        diag = np.diag(x)
        outer = np.outer(x, x.T)
        diag = np.diag((diag - outer))    
        return diag
    
    a = np.apply_along_axis(f, 0, prob)
    b = epis
    df['u_ea2'] = (((a.mean(axis=3) + b.mean(axis=2)).mean(axis=0)).reshape(prob.shape[2]))
    
    return df
    
# util
    
def predict(model, X_val, y_val):
    model_pred = Predictor(model)
    prob = model_pred.predict(X_val)
    
    print(prob.shape)
    
    df = pd.DataFrame()
    df = _addPredictions(df, prob, y_val)
    df = _addSimpleScores(df, prob, prob, prob.shape[1])
    return df

def predict_bbb(model, X_val, y_val):
    model_pred = Predictor(model)
    prob = model_pred.predict_with_uncertainty(X_val, T, dropout=False)
    
    print(prob.shape)
    
    df = pd.DataFrame()
    
    mean_prob = prob.mean(axis=0)
    var_prob = prob.var(axis=0)
    
    
    
    df = _addPredictions(df, mean_prob, y_val)
    df = _addSimpleScores(df, prob, mean_prob, prob.shape[2])
    df = _addAdvancedScores(df, prob, mean_prob, var_prob)
    return df

def predict_mcdropout(model, X_val, y_val):
    model_pred = Predictor(model)
    prob = model_pred.predict_with_uncertainty(X_val, T, dropout=True)
    
    print(prob.shape)
    
    df = pd.DataFrame()
    
    mean_prob = prob.mean(axis=0)
    var_prob = prob.var(axis=0)
    
    df = _addPredictions(df, mean_prob, y_val)
    df = _addSimpleScores(df, prob, mean_prob, prob.shape[2])
    df = _addAdvancedScores(df, prob, mean_prob, var_prob)
    return df
    
def predict_ensamble(models, X_val, y_val):
    model_pred = EnsamblePredictor(models)
    prob = model_pred.predict(X_val)
    
    print(prob.shape)
    
    df = pd.DataFrame()
    
    mean_prob = prob.mean(axis=0)
    var_prob = prob.var(axis=0)
    
    df = _addPredictions(df, mean_prob, y_val)
    df = _addSimpleScores(df, prob, mean_prob, prob.shape[2])
    df = _addAdvancedScores(df, prob, mean_prob, var_prob)
    return df

from tqdm import tqdm
import numpy as np

def bert_predict(model, val_encodings_tf, y_val, T=1):
    preds = []

    steps = 100
    for t in range(T):
        pred_tmp = []
        for i in tqdm(range(0, len(val_encodings_tf.data['input_ids']), steps)):
            j = i + steps
            p_mc = model(input_ids = val_encodings_tf.data['input_ids'][i:j], attention_mask=val_encodings_tf.data['attention_mask'][i:j], training=T>1)
            pres_mc = tf.nn.softmax(p_mc.logits, axis=1).numpy()
            pred_tmp.extend(pres_mc)
        pred_tmp = np.array(pred_tmp)
        preds.append([pred_tmp])
        
    preds = np.array(preds)
    
    print(preds.shape)
    
    mean_prob = preds.mean(axis=0)
    
    df = pd.DataFrame()
    df = _addPredictions(df, mean_prob, y_val, False)
    
    if T <= 1:
        df = _addSimpleScores(df, mean_prob, mean_prob, mean_prob.shape[1])
    else:
        var_prob = preds.var(axis=0)
        
        df = _addSimpleScores(df, preds, mean_prob, preds.shape[2])
        df = _addAdvancedScores(df, preds, mean_prob, var_prob)
    
    return df
    
def bert_predict_en(prob, y_val):
    #model_pred = EnsamblePredictor(models)
    #prob = model_pred.predict(X_val)
    
    df = pd.DataFrame()
    
    mean_prob = prob.mean(axis=0)
    var_prob = prob.var(axis=0)
    
    df = _addPredictions(df, mean_prob, y_val)
    df = _addSimpleScores(df, prob, mean_prob, prob.shape[2])
    df = _addAdvancedScores(df, prob, mean_prob, var_prob)
    return df

    
    
    #prob = []
    #y_val = np.array(y_val)
    
    #for t in tqdm(range(T)):
    #    prob.append([model.predict(X_val.batch(100)).logits])
    
    #prob = np.array(prob)
    
    #df = pd.DataFrame()
    #mean_prob = prob.mean(axis=0)
    
    #df = _addPredictions(df, mean_prob, y_val, False)
    
    #if T <= 1:
    #    df = _addSimpleScores(df, mean_prob, mean_prob, mean_prob.shape[1])
    #else:
    #    var_prob = prob.var(axis=0)
    #    
    #    df = _addSimpleScores(df, prob, mean_prob, prob.shape[2])
    #    df = _addAdvancedScores(df, prob, mean_prob, var_prob)
    
    #return df
    
        
    
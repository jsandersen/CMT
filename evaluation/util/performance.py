import matplotlib.pyplot as plt

from sklearn.metrics import fbeta_score

import numpy as np
import pandas as pd

from contextlib import contextmanager

@contextmanager
def SuppressPandasWarning():
    with pd.option_context("mode.chained_assignment", None):
        yield

def f1_score(df, average='binary'):
    return fbeta_score(df['y_hat'], df['y'], beta=1, average=average)

def f1_score_manually(df, p, u=None, average='binary'):
    l = len(df)
    h = int(l*p)
    
    if u != None:
        # Random Order
        df_sort = df.sort_values(by=[u])
        df_sort = df_sort.reset_index(drop=True)
    else :
        # Uncertainty
        df_sort = df.sample(frac=1).reset_index(drop=True)
    
    if p == 0:
        return f1_score(df_sort, average=average)
    
    df_auto = df_sort[:-h]
    df_manuell = df_sort[-h:]
    with SuppressPandasWarning():
        df_manuell.loc[:, 'y'] = df_manuell['y_hat']
    df_sum = pd.concat([ df_auto,df_manuell ],ignore_index=True)
    
    f_1 = f1_score(df_sum, average=average)
    return f_1

def getMean(array_of_Arrays):
    dd = pd.DataFrame(array_of_Arrays)
    d_mean = dd.mean(axis = 0)
    d_std =  dd.std(axis = 0)
    d_max =  dd.max(axis = 0)
    d_min =  dd.min(axis = 0)
    
    x = d_mean
    
    #print('####')
    #for i in np.arange(0, 0.505, 0.005):
    #    i = round(i, 2)

    #    ab = x[int(len(x)*i)]
    #    diff_to0 = ab - x[0]
    #    print(round(ab, 3), round((diff_to0 / x[0]) * 100, 1) ) 
    
    return d_mean, d_std, d_max, d_min

def plot_conf_line(array, d_mean_r, d_std_r, label, color):
    plt.plot(array, d_mean_r, label=label, color=color)
    plt.plot((d_mean_r[0], 1), color=color, linestyle='dashed')
    plt.fill_between(array, d_mean_r + d_std_r, d_mean_r - d_std_r, color=color, alpha=0.2)
    
def plot_moderation_performance(dfs, u = None, label='Random', color='black', average='binary', eps=200):
    li = []
    range_ = np.arange(0, 1 + (1 / eps), 1 / eps) 
    for df in dfs:
        li_sub = []
        for i in range_:
            i = round(i, 5)
            li_sub.append(f1_score_manually(df, i, u, average=average))
        li.append(li_sub)
    d_mean_m, d_std_m, d_max_m, d_min_m = getMean(li)
    plot_conf_line(range_, d_mean_m, d_std_m, label, color)
    return range_, d_mean_m
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, accuracy_score

# Detection Error
def detectionError(df):
    res = []
    for i in np.arange(0.0, 1, 0.01):
        c_p = df[(df['defect'] == False) & df['p'] <= i]
        c_p = len(c_p) / len(df['defect'] == False)

        c_n = df[(df['defect']) & df['p'] > i]
        c_n = len(c_n) / len(df['defect'])

        r = 0.5 * c_p + 0.5 * c_n
        res.append(r)

    return np.array(res).min()

def print_stats(dfs, average='binary'):
    f1_ss = []
    auc_roc_ss = []
    pr_true_ss = []
    pr_false_ss = []
    mean_true_ss = []
    mean_false_ss = []
    detection_error = []

    for df in dfs:
        f1 = f1_score(df['y_hat'], df['y'], average=average)
        f1_ss.append(f1)

        d_r_ss = detectionError(df)
        detection_error.append(d_r_ss)

        auc_roc = roc_auc_score(df['c_true'], df['p'], average=average)
        auc_roc_ss.append(auc_roc)

        tmp = df[df['defect'] == False]

        pr = average_precision_score(df['c_true'], df['p'], average=average)
        pr_true_ss.append(pr)

        mean_true = tmp['p'].mean()
        mean_true_ss.append(mean_true)

        tmp = df[df['defect']]

        pr = average_precision_score(df['c_false'], -1*df['p'], average=average)
        pr_false_ss.append(pr)

        mean_false = tmp['p'].mean()
        mean_false_ss.append(mean_false)

    def r3(n, r=1):
        return round(n * 100, r) 

    print('F1', r3(np.array(f1_ss).mean(), 3), r3(np.array(f1_ss).std()))
    print('Detection Error', r3(np.array(detection_error).mean()), r3(np.array(detection_error).std()))
    print()
    print('AUC_ROC', r3(np.array(auc_roc_ss).mean()), r3(np.array(auc_roc_ss).std()))
    print('AUC_PR_True', r3(np.array(pr_true_ss).mean()), r3(np.array(pr_true_ss).std()))
    print('AUC_PR_False', r3(np.array(pr_false_ss).mean()), r3(np.array(pr_false_ss).std()))
    print()
    print('MEAN PROB True', r3(np.array(mean_true_ss).mean()), r3(np.array(mean_true_ss).std()))
    print('MEAN PROB False', r3(np.array(mean_false_ss).mean()), r3(np.array(mean_false_ss).std()))
    print('MEAN PROB Range', r3(np.array(mean_true_ss).mean())-r3(np.array(mean_false_ss).mean()))
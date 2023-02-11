import pandas as pd
import numpy as np
import utils_data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
def get_train_test(data_table,test_size=0.25, normalize = False):
    X_train, X_test, y_train, y_test = train_test_split(data_table.drop('subscriber',axis =1)
                                                        , data_table['subscriber']
                                                        , test_size=test_size
                                                        , random_state=27
                                                       ,stratify= data_table['subscriber']
                                                       )
    if normalize:
        X_train, X_test = utils_data.normalize_data( X_train, X_test )
    return X_train, X_test, y_train, y_test
def find_best_results_for_weights(model,param_grid, cv,scoring,X_train, y_train ):
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring=scoring)
    grid_result = grid.fit(X_train, y_train)
    print(grid_result.best_params_)
    return grid_result

def precition_recall_curve(y_val, model_name,probab_val):
    precision, recall, thresholds = precision_recall_curve(y_val, probab_val)
    no_skill = len(y_val[y_val == 1]) / len(y_val)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(recall, precision, marker='.', label=model_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid()
    plt.ylim([0,0.1])
    plt.show()

def get_roc_curve(y_val, model_name,probab_val):
    fpr, tpr, thresholds = roc_curve(y_val, probab_val)
    gmeans = np.sqrt(tpr * (1 - fpr))
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    # plot the roc curve for the model
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label=model_name)
    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid()
    plt.show()
    return thresholds[ix]

def conf_matrix(true,pred):
    ((tn, fp), (fn, tp)) = confusion_matrix(true, pred)
    ((tnr,fpr),(fnr,tpr))= confusion_matrix(true, pred,
            normalize='true')
    return pd.DataFrame([[f'TN = {tn} (TNR = {tnr:1.2%})',
                                f'FP = {fp} (FPR = {fpr:1.2%})'],
                         [f'FN = {fn} (FNR = {fnr:1.2%})',
                                f'TP = {tp} (TPR = {tpr:1.2%})']],
            index=['True 0(No -Subscriber)', 'True 1(Subscriber)'],
            columns=['Pred 0(preds as N0 Subscriber)','Pred 1(pred as Subscriber)'])

def trade_off_plot_fpr_vs_tpr(y_val,probab_val):
    fpr, tpr, thresholds = roc_curve(y_val, probab_val)
    dfplot = pd.DataFrame({'Threshold': thresholds,
                           'False Positive Rate': fpr,
                           'False Negative Rate': 1. - tpr})
    ax = dfplot.plot(x='Threshold', y=['False Positive Rate',
                                       'False Negative Rate'], figsize=(10, 6))
    # zoom in !
    ax.set_xbound(0, 0.0008);
    plt.grid()
    plt.show()

def tune_claasification_with_thershold(pred_prob, thershold):
    return np.where(pred_prob >= thershold, 1, 0)

def get_dict_metrics(y_test, test_classification_preds):
    dict_metrics = {
        'precision_score': precision_score(y_test, test_classification_preds)
        , 'recall_score': recall_score(y_test, test_classification_preds)
        , 'f1_score': f1_score(y_test, test_classification_preds)

    }
    return dict_metrics

def get_metrics_for_validation_and_trainig(dict_preds_and_traget):
    dict_train_val_metrics = { key : {} for key in dict_preds_and_traget }
    for key in dict_preds_and_traget:
        dict_train_val_metrics[key].update(get_dict_metrics(*dict_preds_and_traget[key]))
        dict_train_val_metrics.update({f'{key}_no_skill':get_random_metrics(dict_preds_and_traget[key][0])})
    return pd.DataFrame.from_dict(dict_train_val_metrics, orient='index')

def get_random_metrics(y):
    dy_f= y.to_frame()
    n_positives =dy_f[dy_f['subscriber']==1].shape[0]
    n_negatives =dy_f[dy_f['subscriber']==0].shape[0]
    precision = n_positives/(n_negatives+n_positives)
    recall = 0.5
    f1 = (2*recall*precision)/(recall+precision)
    return {
        'precision_score': precision
        , 'recall_score': recall
        , 'f1_score': f1

    }

def get_thersholds_and_plots( X_train_0, y_train_0,X_val,y_val, model):

    model.fit(X_train_0, y_train_0)
    probab_val = model.predict_proba(X_val)[:, 1]  # taking only the prob of the positives
    threshold = get_roc_curve(y_val, 'xgb', probab_val)
    trade_off_plot_fpr_vs_tpr(y_val, probab_val)
    return threshold
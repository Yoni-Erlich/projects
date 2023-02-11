

import model_utils

def evalute_model(X_train, y_train,X_test,y_test,threshold, model):
    model.fit(X_train, y_train)
    test_classification_preds, training_classification_preds = get_prob(X_test, X_train, model, threshold)
    train_conf_matrix= model_utils.conf_matrix(y_train,training_classification_preds)
    val_conf_matrix= model_utils.conf_matrix(y_test,test_classification_preds)
    dict_preds_and_traget = get_dict_preds_tragets(test_classification_preds, training_classification_preds, y_test,
                                                   y_train)
    metric_df = model_utils.get_metrics_for_validation_and_trainig(dict_preds_and_traget)
    return metric_df, train_conf_matrix.add_suffix('_train'), val_conf_matrix.add_suffix('_val')



def get_dict_preds_tragets(test_classification_preds, training_classification_preds, y_test, y_train):
    dict_preds_and_traget = {
        'train': [y_train, training_classification_preds]
        , 'val': [y_test, test_classification_preds]
    }
    return dict_preds_and_traget


def get_prob(X_test, X_train, model, threshold):
    trainig_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    training_classification_preds = model_utils.tune_claasification_with_thershold(trainig_prob, threshold)
    test_classification_preds = model_utils.tune_claasification_with_thershold(test_prob, threshold)
    return test_classification_preds, training_classification_preds
def plot_eval_results(train_conf, val_conf,metric_df,y_test ,model,X_test  ):
    test_prob = model.predict_proba(X_test)[:, 1]
    display(train_conf)
    display(val_conf)
    display(metric_df)
    model_utils.get_roc_curve(y_test, 'xgb', test_prob)
    model_utils.precition_recall_curve(y_test, 'xgb', test_prob)

def get_main_results(X_train_0, y_train_0, X_val, y_val, threshold, model ):

    metric_df, train_conf, val_conf = evalute_model(X_train_0, y_train_0, X_val, y_val, threshold,
                                                                     model)
    plot_eval_results(train_conf, val_conf, metric_df, y_val, model, X_val)
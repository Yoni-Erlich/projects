
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
import pandas as pd

def get_bias(col_true, col_pred):
    return ((col_true - col_pred)).mean()


def get_median_sales_true(col_true, col_pred):
    return (col_true).median()
def get_median_sales_pred(col_true, col_pred):
    return (col_pred).median()


def get_median_relative_error(col_true, col_pred):
    return abs(100 * (col_true - col_pred) / col_true).median()


def get_std_relative_error(col_true, col_pred):
    return abs(100 * (col_true - col_pred) / col_true).std()
def get_sum_sales(col_true, col_pred):
    return col_true.sum()

def get_metrics(y_train,y_fit ):
    dict_metrics_product = {'RMSE': {}
        , 'MAE': {}
        , 'median_absolute_error': {}
        , 'bias': {}
        , 'median_sales_true': {}
        , 'median_sales_pred': {}
        , 'median_relative_error%': {}
        , 'std_relative_error': {}
        , 'sum_of_sales':{}
                            }
    funcs = {'RMSE': mean_squared_error
        , 'MAE': mean_absolute_error
        , 'median_absolute_error': median_absolute_error
        , 'bias': get_bias
        , 'median_sales_true': get_median_sales_true
        ,'median_sales_pred': get_median_sales_pred
        , 'median_relative_error%': get_median_relative_error
        , 'std_relative_error': get_std_relative_error
        ,'sum_of_sales':get_sum_sales
             }

    for key in dict_metrics_product:
        for col in y_train:
            dict_metrics_product[key].update(
                {col: funcs[key](y_train[col], y_fit[col])}
            )
    df_metrics = pd.DataFrame.from_dict(dict_metrics_product, orient='index').T.sort_values(by='median_relative_error%')
    return df_metrics
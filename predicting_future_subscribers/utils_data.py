import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
def get_row_data(path):
    raw_df = pd.read_csv(path)
    raw_df_index = (
        raw_df
        .drop('Unnamed: 0', axis=1)
        .set_index('id_for_vendor')

    )
    return raw_df_index
def convert_date_cols_to_date_type(raw_df_index):
    return raw_df_index.assign(device_timestamp=lambda df: pd.to_datetime(df['device_timestamp'], format='%Y-%m-%d %H:%M:%S')
                        , install_date=lambda df: pd.to_datetime(df['install_date'], format='%Y-%m-%d %H:%M:%S')
                        ,
                        subscription_date=lambda df: pd.to_datetime(df['subscription_date'], format='%Y-%m-%d %H:%M:%S')

                        )

def add_time_diff_from_installation_and_from_subscribtion(raw_df_index):
    return (raw_df_index
    .assign(time_from_install_to_last_usage=lambda df: (df['device_timestamp'] - df['install_date']).astype(
        'timedelta64[m]')/60
            , time_from_install_to_subscribe=lambda df: (df['subscription_date'] - df['install_date']).astype(
            'timedelta64[m]')/60
            )
     )

def filter_usage_beyond_24_hours_from_installment(df):
    filtered_df =df[df['time_from_install_to_last_usage']<=24]
    print(f'data loss from filter data pass 24h of installment {1-filtered_df.shape[0]/df.shape[0]}')
    return filtered_df

def filter_usage_before_installment(df):
    filtered_df =df[df['time_from_install_to_last_usage']>=0]
    print(f'data loss from filter_usage_before_installment {1-filtered_df.shape[0]/df.shape[0]}')
    return filtered_df

def filter_subscription_less_24h_from_intallment(df):
    value_for_nan = round(df['time_from_install_to_subscribe'].max() + 10)
    filtered_df = (df
                            .assign(
        time_from_install_to_subscribe=lambda df: df['time_from_install_to_subscribe'].fillna(value_for_nan))

                            .pipe(lambda df: df[df['time_from_install_to_subscribe'] > 24])
                            .replace(value_for_nan, np.NaN)
                            )
    print(f'data loss from filter data pass for subscription less than 24h from installation {1 - filtered_df.shape[0] / df.shape[0]}')
    return filtered_df
def filter_subscription_more_24h_from_intallment(df):
    filtered_df = df[df['time_from_install_to_subscribe'] > 24]
    print(f'data loss due to filter of subscription more 24h from intallment {1 - filtered_df.shape[0] / df.shape[0]}')
    return filtered_df

def read_and_basic_clean_filter_data(data_path):
    raw_df = get_row_data(data_path)
    raw_df_index_filter = (raw_df
                           .pipe(lambda df: convert_date_cols_to_date_type(df))
                           .pipe(lambda df: add_time_diff_from_installation_and_from_subscribtion(df))
                           .pipe(lambda df: filter_usage_beyond_24_hours_from_installment(df))
                           .pipe(lambda df: filter_subscription_less_24h_from_intallment(df))
                           .pipe(lambda df: filter_usage_before_installment(df))



                           )
    return raw_df_index_filter

def get_device_type_one_hot(raw_df_index_filter):
    devices_per_user = (raw_df_index_filter
                        .groupby('id_for_vendor')
                        [['device']]
                        .first()
                        .assign(device=lambda df: df['device'].str.lower())
                        ['device']
                        .apply(lambda x: 'ipad' if 'ipad' in x else x)
                        .apply(lambda x: 'iphone' if 'iphone' in x else x)
                        .apply(lambda x: 'ipod' if 'ipod' in x else x)
                        .to_frame()

                        )
    devices_per_user_one_hot = pd.get_dummies(devices_per_user.device, prefix='device')
    return devices_per_user_one_hot

def get_gdp_data_per_user(df):
    # i have donwload the data from here :
    # https://ourworldindata.org/grapher/gdp-per-capita-worldbank

    country_gdp_per_capita = (pd.read_csv('gdp-per-capita-worldbank.csv')
                              .query('Year ==2020')
                              [['Entity', 'GDP per capita, PPP (constant 2017 international $)']]
                              .rename(
        columns={'Entity': 'country', 'GDP per capita, PPP (constant 2017 international $)': 'GDP_PC'})
                              .assign(GDP_PC=lambda df: pd.qcut(df['GDP_PC'], q=4, labels=[0, 1, 2, 3]))
                              )

    dict_gdp= get_dict_country_gdp(country_gdp_per_capita)
    users_gdp = (df
    .reset_index()
    [['id_for_vendor', 'country']]
    .drop_duplicates(subset="id_for_vendor")
    .assign(gdp_value=lambda df: df['country'].map(dict_gdp))
    .set_index('id_for_vendor')[['gdp_value']]
    )
    return users_gdp

def get_delta_time_between_session_stats(raw_df_index_filter):
    df_w_delta_seconds = (raw_df_index_filter
                          .sort_values(by='device_timestamp')
                        .groupby(['id_for_vendor', 'app_session_id'])
                        [['device_timestamp']]
                        .apply(lambda g: g - g.shift())
                        .assign(delta_s_between_sessions=lambda df: df[['device_timestamp']].astype('timedelta64[s]'))
                          .dropna()
                          .groupby('id_for_vendor')
                          [['delta_s_between_sessions']]
                          .agg(delta_s_between_sessions_mean=('delta_s_between_sessions', 'mean')
                               , delta_s_between_sessions_median=('delta_s_between_sessions', 'median')
                               , delta_s_between_sessions_std=('delta_s_between_sessions', 'std')

                               ).fillna(0)

     )
    return df_w_delta_seconds

def get_dict_country_gdp(country_gdp_per_capita):
    dict_gdp = country_gdp_per_capita.set_index('country').to_dict()['GDP_PC']
    # updating some nan's according quich seach in google
    dict_gdp['Yemen'] = 0
    dict_gdp['Venezuela'] = 0
    dict_gdp['Hong Kong (China)'] = dict_gdp['Hong Kong']
    dict_gdp['Taiwan'] = 3
    dict_gdp['Bosnia & Herzegovina'] = 2
    dict_gdp['Eritrea'] = 0
    dict_gdp['Myanmar (Burma)'] = dict_gdp['Myanmar']
    dict_gdp['Macedonia'] = 2
    dict_gdp['Congo - Brazzaville'] = 0
    return dict_gdp

def get_features_from_numeric_data_with_agg(raw_df_index_filter):
    return (raw_df_index_filter
    .groupby('id_for_vendor')
    .agg(total_usage_time=('usage_duration', sum)
         , num_unique_features=('feature_name', lambda x: x.nunique())
         , num_of_unique_sessions=('app_session_id', lambda x: x.nunique())
         , accepted_mean=('accepted', 'mean')
         , actions_count=('accepted', 'count')
         , subscriber=('subscriber', 'mean')
         )

)

def get_class_populations(df):
    return df[df['subscriber']==1], df[df['subscriber']==0]

def plot_box_plot(df):
    subscribers, none_subscribers = get_class_populations(df)
    for col in df.columns:
        f, axes = plt.subplots(1, 3, figsize= (12,7))
        sns.boxplot(x=subscribers[col],ax=axes[0])
        axes[0].legend(['Subscribers'])
        sns.boxplot(x=none_subscribers[col], ax=axes[1])
        axes[1].legend(['Non - subscribers'])
        sns.boxplot(x=df[col], ax=axes[2])
        axes[2].legend(['Full population'])
        f.suptitle(col)
        plt.show()

def get_class_ratio(df):
    return df[df['subscriber'] == 1].shape[0]/df[df['subscriber'] == 0].shape[0]

def get_mean_action_features_per_sesssion (raw_df_index_filter):
    mean_action_features_per_sesssion = (
        raw_df_index_filter
        .groupby(['id_for_vendor', 'app_session_id'])
        .agg(action_per_session=('accepted', 'count')

             )
        .groupby('id_for_vendor')
        .mean()

    )
    return mean_action_features_per_sesssion

def clean_outliers_set_of_rules(X_train_and_validation):
    # this set of rules is without viewing the test data
    # set of. filters :
    X_train_and_validation = X_train_and_validation[X_train_and_validation['total_usage_time'] < 8]
    X_train_and_validation = X_train_and_validation[X_train_and_validation['actions_count'] < 100]
    X_train_and_validation = X_train_and_validation[X_train_and_validation['delta_s_between_sessions_mean'] < 500]
    X_train_and_validation = X_train_and_validation[X_train_and_validation['delta_s_between_sessions_median'] < 500]
    X_train_and_validation = X_train_and_validation[X_train_and_validation['delta_s_between_sessions_std'] < 400]
    X_train_and_validation = X_train_and_validation[X_train_and_validation['action_per_session'] < 30]
    X_train_and_validation = X_train_and_validation[X_train_and_validation['normalize_usage_time'] < 250]
    X_train_clean = X_train_and_validation.drop('subscriber', axis=1)
    y_train_clean = X_train_and_validation['subscriber']
    return X_train_and_validation, X_train_clean, y_train_clean

def normalize_data(X_train, X_test):
    col_not_to_scale = ['gdp_value', 'device_ipad', 'device_iphone', 'device_ipod']

    col_to_scale = ['total_usage_time', 'num_unique_features', 'num_of_unique_sessions',
                    'accepted_mean', 'actions_count',
                    'delta_s_between_sessions_mean', 'delta_s_between_sessions_median',
                    'delta_s_between_sessions_std', 'action_per_session', 'normalize_usage_time']

    ss = preprocessing.StandardScaler()
    df_scaled_train = pd.DataFrame(ss.fit_transform(X_train[col_to_scale]), columns=X_train[col_to_scale].columns,
                             index=X_train[col_to_scale].index)
    df_scaled_test = pd.DataFrame(ss.transform(X_test[col_to_scale]), columns=X_test[col_to_scale].columns,
                                   index=X_test[col_to_scale].index)

    X_train_scaled = df_scaled_train.join(X_train[col_not_to_scale])
    X_test_scaled = df_scaled_test.join(X_test[col_not_to_scale])
    return X_train_scaled, X_test_scaled
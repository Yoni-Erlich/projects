
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess


def plot_corr_between_stores(train):
    df_train = train[["store_id", "sales"]]
    df_train["ind"] = 1
    df_train["ind"] = df_train.groupby("store_id").ind.cumsum().values
    df_corr_stores_sales = pd.pivot(df_train, index="ind", columns="store_id", values="sales").corr()
    mask = np.triu(df_corr_stores_sales.corr())
    plt.figure(figsize=(20, 20))
    sns.heatmap(df_corr_stores_sales,
                annot=True,
                fmt='.1f',
                cmap='coolwarm',
                square=True,
                mask=mask,
                linewidths=1,
                cbar=False)
    plt.title("Correlations among stores", fontsize=20)
    plt.show()
    return  df_corr_stores_sales
def get_zero_selling(train):

    zero_selling_products = (train
                             .groupby(["store_id", "sku_category"])
                             .sales
                             .sum()
                             .reset_index()
                             .sort_values(["sku_category", "store_id"])
                             .pipe(lambda df: df[df.sales == 0])

                             )
    return zero_selling_products


def removd_zero_selling(train):
    shape_before = train.shape[0]
    zero_selling_products = get_zero_selling(train)
    outer_join = train.merge(zero_selling_products.drop("sales", axis=1), how='outer', indicator=True)
    train = outer_join[~(outer_join._merge == 'both')].drop('_merge', axis=1)
    print(f'total data removed due to zero selling ={1-train.shape[0]/shape_before}')
    return train

def removd_zero_selling_test(test, zero_selling_train):
    shape_before = test.shape[0]
    outer_join = test.merge(zero_selling_train.drop("sales", axis=1), how='outer', indicator=True)
    test = outer_join[~(outer_join._merge == 'both')].drop('_merge', axis=1)
    print(f'total data removed due to zero selling ={1-test.shape[0]/shape_before}')
    return test

def get_sku_sales_resolution(train):
    store_sales = train.copy()
    store_sales['date'] = store_sales.date.dt.to_period('D')
    store_sales = store_sales.set_index(['store_id', 'sku_category', 'date']).sort_index()
    sku_sales = (
        store_sales
         .groupby(['sku_category', 'date'])
        .mean()
        .unstack(['sku_category'])
    )
    return sku_sales

def get_test_X1_X2(test,train,lin_order=1):
    sku_resolution_test = get_sku_sales_resolution(test)

    dp = DeterministicProcess(index=sku_resolution_test.index, order=lin_order)
    X_1 = dp.in_sample()

    X_2 = sku_resolution_test.stack()
    X_2 = (X_2.pipe(lambda df: create_date_features_for_seasonality(df.reset_index()))
                .pipe(lambda df: add_sales_amount_feature(df, train))

                )
    return X_1, X_2

def get_sales_amount_feature(train):
    df_feature=  (train.groupby(['sku_category'])
    .sum()
    [['sales']]
    .sort_values(by='sales')
    .assign(sales_amount=lambda df: pd.qcut(df['sales'], q=4, labels=[0, 1, 2, 3]))
    [['sales_amount']]
    )
    return df_feature.to_dict()['sales_amount']
def get_per_sku_per_store_features(train, stores):
    store_sales = train.copy()
    store_sales_w_features = (
        store_sales
        .reset_index()
        .merge(stores.reset_index(), on='store_id')
        .set_index(['store_id', 'sku_category', 'date']).sort_index()

    )

    df_str = store_sales_w_features.select_dtypes(include='object')
    enc = OrdinalEncoder(categories=[
        df_str[col].unique()
        for col in df_str.columns
    ])

    store_sales_w_features = pd.concat([
        pd.DataFrame(enc.fit_transform(df_str), columns=df_str.columns, index=df_str.index),
        store_sales_w_features.select_dtypes(exclude='object')
    ], axis=1)
    return store_sales_w_features

def add_sales_amount_feature(df,train):
    sales_amount = get_sales_amount_feature(train)
    return df.assign(sale_amount=lambda df: df['sku_category'].map(sales_amount))

def get_y_X_for_hybrid_model(sku_sales, train, oreder =1):

    y_sku_sales = sku_sales.loc[:, 'sales']
    dp = DeterministicProcess(index=y_sku_sales.index, order=oreder)
    X_1 = dp.in_sample()
    X_2 = sku_sales.drop('sales', axis=1).stack()  # onpromotion feature
    X_2 = (X_2.pipe(lambda df: create_date_features_for_seasonality(df.reset_index()))
            .pipe(lambda df:add_sales_amount_feature(df, train) )

           )
    return y_sku_sales,X_1, X_2

def get_train_val(y_sku_sales,time_to_train_upper_limit,time_for_val_down_limit, X_1, X_2  ):
    y_train, y_valid = y_sku_sales[:time_to_train_upper_limit], y_sku_sales[time_for_val_down_limit:]
    X1_train, X1_valid = X_1[: time_to_train_upper_limit], X_1[time_for_val_down_limit:]
    X2_train, X2_valid = X_2.loc[:time_to_train_upper_limit], X_2.loc[time_for_val_down_limit:]
    return y_train, y_valid, X1_train, X1_valid, X2_train, X2_valid

def create_date_features_for_seasonality(df):
    df['month'] = df.date.dt.month.astype("int8")
    df['day_of_month'] = df.date.dt.day.astype("int8")
    df['day_of_year'] = df.date.dt.dayofyear.astype("int16")
    df['week_of_month'] = (df.date.apply(lambda d: (d.day-1) // 7 + 1)).astype("int8")
    df['week_of_year'] = (df.date.dt.weekofyear).astype("int8")
    df['day_of_week'] = (df.date.dt.dayofweek + 1).astype("int8")
    # df['year'] = df.date.dt.year.astype("int32")
    df["is_wknd"] = (df.date.dt.weekday // 4).astype("int8")
    df["quarter"] = df.date.dt.quarter.astype("int8")
    # 0: Winter - 1: Spring - 2: Summer - 3: Fall
    df["season"] = np.where(df.month.isin([12,1,2]), 0, 1)
    df["season"] = np.where(df.month.isin([6,7,8]), 2, df["season"])
    df["season"] = pd.Series(np.where(df.month.isin([9, 10, 11]), 3, df["season"])).astype("int8")
    return df.set_index('date')

def plot_stats(df, column, ax, color, angle):
    """ PLOT STATS OF DIFFERENT COLUMNS """
    count_classes = df[column].value_counts()
    ax = sns.barplot(x=count_classes.index, y=count_classes, ax=ax, palette=color)
    ax.set_title(column.upper(), fontsize=18)
    for tick in ax.get_xticklabels():
        tick.set_rotation(angle)
def plot_stats_of_all_store_features(stores):
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(20, 40))
    plot_stats(stores, "geo", axes[0], "mako_r", 45)
    plot_stats(stores, "province", axes[1], "rocket_r", 45)
    plot_stats(stores, "type", axes[2], "magma", 0)
    plot_stats(stores, "group", axes[3], "viridis", 0)
    plt.show()


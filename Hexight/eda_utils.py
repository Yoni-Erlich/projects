
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def plot_sum_of_sales(train):
    (train.query('date.dt.year>2016').groupby(['sku_category'])
     .sum()
     [['sales']]
     .sort_values(by='sales')
     .plot.bar()
     )
    plt.grid()

def plot_daily_sales_per_sku(train):
    train_w_diff_time_periods = (train
            .query('date.dt.year >2015')
            .groupby(['sku_category', 'date'])
            .mean()
            [['sales']]
            .reset_index()
            .assign(dayofweek=lambda df: df.date.dt.dayofweek
                    , week=lambda df: df.date.dt.week
                    , dayofyear=lambda df: df.date.dt.dayofyear
                    , year=lambda df: df.date.dt.year
                    )

            )
    for sku in train_w_diff_time_periods['sku_category'].unique():
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle(f'SKU = {sku}')
        seasonal_plot(train_w_diff_time_periods.query(f'sku_category =={sku}'), 'sales', 'year', 'dayofyear', ax=ax1)
        seasonal_plot(train_w_diff_time_periods.query(f'sku_category =={sku}'), 'sales', 'year', 'week', ax=ax2)
        # plt.title(f'SKU = {sku}')
        plt.tight_layout()
        plt.show()

def seasonal_plot(X, y, period, freq, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    palette = sns.color_palette("husl", n_colors=X[period].nunique(),)
    ax = sns.lineplot(
        x=freq,
        y=y,
        hue=period,
        data=X,
        ci=False,
        ax=ax,
        palette=palette,
        legend=False,
    )
    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    for line, name in zip(ax.lines, X[period].unique()):
        y_ = line.get_ydata()[-1]
        ax.annotate(
            name,
            xy=(1, y_),
            xytext=(6, 0),
            color=line.get_color(),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            size=14,
            va="center",
        )
    return ax


def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = pd.Timedelta("1Y") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax


def plot_moving_avg(train,moving_avg = 120):
    for sku in train['sku_category'].unique():
        df_moving_avg = (
            train.query(f'sku_category== {sku}')
            .groupby(['sku_category', 'date'])
            [['sales']]
            .mean()
            .rolling(moving_avg)
            .mean()
            .reset_index()
            [['sales']]
            .plot()
        )
        plt.title(f'SKU = {sku}, moving_avg ={moving_avg} days')
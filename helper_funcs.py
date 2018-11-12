# Data packages
import numpy as np
import pandas as pd

# Visualization packages
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ML packages
from statsmodels.tsa.arima_model import ARMA
from sklearn.metrics import mean_squared_error

# Helper function definitions for ClimaCell New Technologies homework assignment

def plot_autocorr(series, lags = 100, alpha = 0.05, zero = True, figsize = (15, 8)):
    """Plot autocorrelation function and partial autocorrelation function for a time series."""
    fig, ax = plt.subplots(figsize = figsize, nrows = 2, sharex = True)
    sm.graphics.tsa.plot_acf(series, lags = lags, alpha = alpha, ax = ax[0], zero = zero);
    ax[0].set_ylabel('ACF')
    sm.graphics.tsa.plot_pacf(series, lags = lags, alpha = alpha, ax = ax[1], zero = zero);
    ax[1].set_xlabel('Lag')
    ax[1].set_ylabel('PACF')
    for x in ax:
        x.grid()


def plot_ci(x, y, yerr_min, yerr_max, ax = None, **kwargs):
    """Plot confidence interval around values."""
    ax.fill_between(x, y - yerr_min, y + yerr_max, **kwargs)
    return ax


def ARMA_train_forecast(series, cutoff_date, diffs = [], order = (0, 0), freq = 'H'):
    """Split a time series into training and test sets, train an ARMA model, and use it to
       recover training data and forecast test data."""
    
    # Split time series into training and test set on cutoff_date
    train, test = series[series.index.date < cutoff_date], series[series.index.date == cutoff_date]

    # Take differences of training series as specified in diffs, and drop undefined values
    diff_train = train.copy()
    for d in diffs:
        diff_train = diff_train.diff(d)
    diff_train = diff_train.dropna()

    # Calculate additive term to effectively "un-difference" the output of the model
    undiff_term = pd.Series(np.zeros(len(series)), index = series.index)
    diff_series = series.copy()
    for d in diffs:
        undiff_term += diff_series.shift(d)
        diff_series = diff_series.diff(d)
    undiff_term = undiff_term.dropna()

    # Train ARMA model
    model = ARMA(diff_train, order = order, freq = freq).fit(disp = False)

    # Find model's prediction of training data
    predict = (model.predict() + undiff_term.loc[train.index]).dropna()

    # Find model's prediction of test data (including standard error and confidence interval)
    forecast = list(model.forecast(len(test)))
    forecast_df = pd.DataFrame(np.concatenate([np.stack(forecast[:2], axis = 1), forecast[2]], axis = 1),
                               columns = ['forecast', 'stderr', 'conf_int_min', 'conf_int_max'],
                               index = test.index)
    forecast_df.forecast += undiff_term.loc[test.index]
    
    return model, train, test, predict, forecast_df


def rmse(y_true, y_pred):
    """Compute the root mean squared error (RMSE), given actual and predicted values."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def plot_ARMA(train, test, predict, forecast_df, plot_hours = 24, figsize = (10, 8)):
    """Plot training and test sets along with respective predicted and forecast values with confidence interval.
       Also, output root mean squared error (RMSE) for both training and test sets."""
    fig, ax = plt.subplots(figsize = figsize)

    pd.concat([train, test])[-plot_hours:].plot(ax = ax, color = 'blue', label = 'Actual');
    predict[-24 * 1:].plot(ax = ax, color = 'green', label = 'Predicted');
    forecast_df.forecast.plot(ax = ax, color = 'red', label = 'Forecast');
    plot_ci(test.index, forecast_df.forecast, -forecast_df.conf_int_min, forecast_df.conf_int_max, color = 'red', alpha = 0.2, ax = ax, label = 'Confidence Interval')
    
    ax.legend()
    ax.grid()
    ax.set_title('Temperature over {} Hours'.format(plot_hours))
    ax.set_xlabel('Time');
    ax.set_ylabel('Temperature ($^{\circ}$ Fahrenheit)');
    
    print("RMSE (train): {}\nRMSE (test): {}".format(rmse(train.loc[predict.index], predict), rmse(test, forecast_df.forecast)))
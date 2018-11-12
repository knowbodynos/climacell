# ClimaCell New Technologies Homework

***
# Table of Contents

1. [Goals](./climacell_new_technologies_homework.ipynb#goals)
2. [Summary](./climacell_new_technologies_homework.ipynb#summary)
3. [Import Packages](./climacell_new_technologies_homework.ipynb#import)
4. [Load Data](./climacell_new_technologies_homework.ipynb#load)
   1. [Load ASOS CID data](./climacell_new_technologies_homework.ipynb#CID)
   2. [Set correct data types](./climacell_new_technologies_homework.ipynb#types)
   3. [Handle duplicate and missing data](./climacell_new_technologies_homework.ipynb#missing)
5. [Exploratory Data Analysis](./climacell_new_technologies_homework.ipynb#eda)
   1. [Plot data](./climacell_new_technologies_homework.ipynb#plot)
   2. [Autocorrelation](./climacell_new_technologies_homework.ipynb#autocorr)
6. [Model Building](./climacell_new_technologies_homework.ipynb#model)
   1. [ARMA Model](./climacell_new_technologies_homework.ipynb#arma)
7. [Results](./climacell_new_technologies_homework.ipynb#results)
   1. [Forecast](./climacell_new_technologies_homework.ipynb#forecast)

***
# Goals <a name="goals"></a>

1. Design  a simple machine learning/statistics-based forecasting model using this data. (No
physical/meteorological modeling.) The goal is to predict the temperature every hour for the day of 2018-06-01 (i.e. midnight to midnight in local time), trained only on data prior to that day.
2. Are there large data gaps? How big? How are you handling this?
3. Is the data on a regular time grid? If not, how are you handling this?
4. Make a plot showing the results of your model, compared to the true temperature data for 2018-06-01.
5. Produce an aggregate error metric to assess your model's performance -- how's it do?
6. Please write a short blurb explaining your methodology/assumptions/etc. Also please discuss other
methodologies that could be investigated, and how they might improve upon your results.

***
# Summary <a name="summary"></a>

I first downloaded air temperature (in Fahrenheit) data from [https://mesonet.agron.iastate.edu/request/download.phtml](https://mesonet.agron.iastate.edu/request/download.phtml) for the \[CID\] Cedar Rapids ASOS station between June 1, 2008 and June 2, 2018 (10 years). After loading the data, I found that there were over 200,000 missing data points, and I dropped these. In addition, the data was collected at uneven intervals: some observations were taken at the same time, and others were taken between 1 minute and 1.5 days apart. To simultaneously smooth over some of the uneven sampling and create a regular hourly time grid, I resampled the data by averaging over 1 hour periods. Then, I addressed the missing values by iteratively filling them in with values from the previous day until they were all filled. At this point, I had a complete data set.

To begin my analysis, I simply plotted the time series, along with a monthy and yearly rolling average. The monthly averages showed a clear seasonality, while the yearly averages showed a stationary overall trend. I also plotted centered temperatures for the average month, reinforcing the seasonality from earlier, and the centered temperatures for the average day, which showed a new daily seasonality. Because this is a univariate time series, I then inspected its autocorrelation (ACF) and partial autocorrelation (PACF) functions at different lag times. I found that much of the autocorrelation was explained by a seasonality of 1 year, as predicted earlier. Differencing by 1 year, and inspecting the new ACF and PACF, I found that just about all of the remaining autocorrelation was explained by a lag of 1 hour. Differencing again by 1 hour, very little autocorrelation remained except at the 2 hour and 1 day lags, but not enough to require another full differencing. Incorporating the 1 day lag into an autoregressive (AR) or moving averages (MA) model would be too memory-intensive, so I looked only at the 2 hour lag. Because both the PACF and ACF plots had a small peak at the 2 hour lag, this suggested an ARMA(2,2) model.

As proof of concept, I fit an ARMA(2,2) model to a training set with a random cutoff date, and used it to forecast the temperatures for the next day. I also used the model to predict its training data. Comparing these to the actual temperature values, I computed the root mean squared error (RMSE) metric for both training and test data and found both to be less than 3 degrees F. In addition, the actual temperatures fell well within the confidence interval of the forecast values.

In order to get a better idea of my model accuracy, I plotted the RMSE for both training and test data as I varied the training set cutoff date. I found that the training RMSE stayed constant at ~2.5 degrees F, while the test RMSE fluctuated, but stayed under 6 degrees F.

Finally, I forecast the temperature at the Cedar Rapids, Iowa (CID) ASOS weather station for 2018-06-01. I found that both the training and test RMSE were ~2.5 degrees F, and that the actual values stayed well within the confidence interval of the forecast.

I think that if it were computationally feasible to implement an ARMA(24,24), using the ACF and PACF peaks around lag 24, I might have been able to achieve even better results. Also, if there were more sources of data available, such as nearby weather stations, CO2 levels, etc. I could implement a multivariate model that might improve performance.
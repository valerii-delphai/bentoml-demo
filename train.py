import lightgbm as lgbm
import numpy as np
import pandas as pd
import pickle

# separate feature spaces for each model (casual and registered users)
features_casual = ['temp', 'hum', 'windspeed', 'hr', '3_days_sum_casual',
            'rolling_mean_12_hours_casual','season', 'yr', 'mnth',  
            'day_type', 'weathersit', 'CasualHourBins', 'weekday']

features_registered = ['temp', 'hum', 'windspeed', 'hr', '3_days_sum_registered',
            'rolling_mean_12_hours_registered', 'season', 'yr', 'mnth',  
            'day_type', 'weathersit', 'RegisteredHourBins', 'weekday']

# optimized models parameters
parameters_casual = {
  "objective": "regression",
  "metric": "rmse",
  "boosting_type": "gbdt",
  "verbosity": -1,
  "lambda_l1": 0.2019055894080857,
  "lambda_l2": 3.5275169933928286e-07,
  "num_leaves": 110,
  "feature_fraction": 0.92,
  "bagging_fraction": 1.0,
  "bagging_freq": 0,
  "min_child_samples": 20,
  #"num_boost_round": 117
}

parameters_registered = {
  "objective": "regression",
  "metric": "rmse",
  "boosting_type": "gbdt",
  "verbosity": -1,
  "lambda_l1": 0.010882827930218712,
  "lambda_l2": 0.2708162972907513,
  "num_leaves": 71,
  "feature_fraction": 0.8,
  "bagging_fraction": 0.7260429650751228,
  "bagging_freq": 2,
  "min_child_samples": 20,
  #"num_boost_round": 147
}

def prepare_data(data):
    # merging 3 and 4 categories of weather situation feature
    data['weathersit'].replace(to_replace=4, value=3, inplace=True)
    
    # creating 'day_type' feature
    ### 2 - working day
    ### 1 - weekend
    ### 0 - holiday

    # if `holiday` is equal to 1, we encode observation as 0, otherwise - 2
    data['day_type'] = np.where(data['holiday'] == 1, 0, 2)
    # if `weekday` is saturday or sunday, encode observation as 1, otherwise leave it as it is
    data['day_type'] = np.where((data['weekday'] == 6) | (data['weekday'] == 0), 1, data['day_type'])
    
    # binning hour into 'RegisteredHourBins' feature
    bins = np.array([1.5, 5.5, 6.5, 8.5, 16.5, 18.5, 20.5, 22.5])
    labels = np.arange(len(bins)-1)
    remap_labels = {0: 0, 1: 1, 2: 5, 3: 3, 4: 6, 5: 4, 6: 2}
    data['RegisteredHourBins'] = pd.cut(data['hr'], bins=bins, labels=labels).fillna(0).astype(int)
    data['RegisteredHourBins'] = data['RegisteredHourBins'].map(remap_labels)

    # binning hour into 'CasualHourBins' feature
    bins = np.array([7.5, 8.5, 10.5, 17.5, 19.5, 21.5])
    labels = np.arange(len(bins)-1)
    remap_labels = {0: 0, 1: 2, 2: 4, 3: 3, 4: 1}
    data['CasualHourBins'] = pd.cut(data['hr'], bins=bins, labels=labels).fillna(0).astype(int)
    data['CasualHourBins'] = data['CasualHourBins'].map(remap_labels)
    
    # rolling mean features
    data['rolling_mean_12_hours_casual'] = data['casual'].rolling(min_periods=1, window=12).mean()
    data['rolling_mean_12_hours_registered'] = data['registered'].rolling(min_periods=1, window=12).mean()

    # sum of shifted values features
    data['3_days_sum_casual'] = data['casual'].shift(24, fill_value=0) +\
                                data['casual'].shift(48, fill_value=0) +\
                                data['casual'].shift(72, fill_value=0)
    data['3_days_sum_registered'] = data['registered'].shift(24, fill_value=0) +\
                                    data['registered'].shift(48, fill_value=0) +\
                                    data['registered'].shift(72, fill_value=0)
    return data
    
if __name__=="__main__":
    # read the hourly base data while parsing dates
    data = pd.read_csv("hour.csv", 
                        parse_dates=[1])
    # prepare data
    X = prepare_data(data)
    
    # get separate training sets
    train_casual= lgbm.Dataset(X[features_casual], X['casual'])
    train_registered = lgbm.Dataset(X[features_registered], X['registered'])

    # train models with best parameters
    model_1 = lgbm.train(parameters_casual, train_set=train_casual, num_boost_round=117)
    model_2 = lgbm.train(parameters_registered, train_set=train_registered, num_boost_round=147)


    # save models to drive
    with open('./src/model_casual.pkl', 'wb') as file:
        pickle.dump(model_1, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./src/model_registered.pkl', 'wb') as file:
        pickle.dump(model_2, file, protocol=pickle.HIGHEST_PROTOCOL)

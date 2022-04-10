# bentoml-demo
This is repository for deployment of my previous [pet-project](https://github.com/valeriich/midterm-project) using the BentoML framework.
Project was about forecasting hourly demand in the bike rentals service.
Two LightGBM regression models were developed to predict demand for casual and registered users separately.
Sum of these two models predictions gives us the hourly forecast.

### Repository structure
* `hour.csv` - [dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset) with raw data
* `train.py` - script to preprocess data, train models and save them in pickle format
* `test.py` - script to test bentoml service with a data sample; true value for this sample is 241
* `pyproject.toml` - dependencies to activate virtual environment with poetry
* `poetry.lock` - poetry file with the list of exact package versions
* `bentofile.yaml` - bentoml configuration file *(left empty for now)*
* `src/main.py` - script which defines bentoml service API
* `scr/save.py` - script for packaging bentoml service
* `model_casual.pkl`, `model_registered.pkl` - pickled trained LightGBM models

### How to put it on service
To run this service execute following commands:
* `git clone https://github.com/valerii-delphai/bentoml-demo`
* `cd bentoml-demo`
* `poetry shell`
* `bentoml build`
* `bentoml serve BikeRentalsPredictionService:latest`

### To test a data sample run commands in new terminal window
* `cd bentoml-demo`
* `python3 test.py`

A data sample in JSON format looks like this:

        {"3_days_sum_casual": 44.0,
        "3_days_sum_registered": 499.0,
        "CasualHourBins": 1.0,
        "RegisteredHourBins": 4.0,
        "day_type": 2.0,
        "hr": 20.0,
        "hum": 0.49,
        "mnth": 11.0,
        "rolling_mean_12_hours_casual": 26.916666666666668,
        "rolling_mean_12_hours_registered": 272.6666666666667,
        "season": 4.0,
        "temp": 0.32,
        "weathersit": 1.0,
        "weekday": 1.0,
        "windspeed": 0.2537,
        "yr": 1.0,
        "holiday": 0.0}

Expected output is 243 bikes.

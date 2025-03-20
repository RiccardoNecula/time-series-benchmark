#contenitore di hparams

Auto_arima_params = {
    "seasonal" : True,
    "stepwise" : True,
    "trace" : True,
    "suppress_warning" : True,
    "test": "adf"
}


LSTM_grid_params = {
    'unit1_min': 32,
    'unit1_max': 256,
    'unit1_step': 32,

    'unit2_min': 32,
    'unit2_max': 128,
    'unit2_step': 32,

    'dropout_min': 0.1,
    'dropout_max': 0.3,
    'dropout_step': 0.1,

    'learning_rate_min': 0.01,
    'learning_rate_max': 0.1,
    'learning_rate_step': 0.03,
}


ETS_smoothing_params = {
    "trend": ["add", "mul"],
    "seasonal": ["add", "mul"],
    "seasonal_periods": [7],
    "damped_trend": [True, False]
}


Prophet_grid_params = {
    'changepoint_prior_scale': [0.01, 0.05],
    'seasonality_prior_scale': [0.2, 0.3],
    'weekly_seasonality': [True, False]
}


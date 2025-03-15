#contenitore di hparams

Auto_arima_params = {
    "seasonal" : False,
    "stepwise" : True,
    "trace" : True,
    "suppress_warning" : True
}


LSTM_grid_params = {
    'unit1_min': 32,
    'unit1_min': 256,
    'unit1_step': 32,

    'unit2_min': 32,
    'unit2_max': 128,
    'unit2_step': 32,

    'dropout_min': 0.1,
    'dropout_max': 0.3,
    'dropout_step': 0.1,

    'learning_rate_min': 0.01,
    'learning_rate_max': 0.1,
    'learning_rate_step': 0.5,
}


ETS_smoothing_params = {
    "trend": ["add", "mul", None],
    "seasonal": ["add", "mul", None],
    "seasonal_periods": [12],
    "damped_trend": [True, False]
}


Prophet_grid_params = {
    'changepoint_prior_scale': [0.01, 0.1, 0.5, 1.0, 10.0],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'weekly_seasonality': [True, False]
}


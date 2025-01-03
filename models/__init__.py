import tensorflow as tf

# Configura i log per ridurre il rumore
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

__all__ = ['Sequential',
           'Dense',
           'LSTM',
           'Dropout',
           'Adam',
           'tf'
           ]

# Debug config message
print("Keras con TensorFlow è stato configurato correttamente.")


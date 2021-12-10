# Examples and Usage of TimeWarPy

## Preprocessing Univariate Data for Recurrent Neural Network Training

With less than 30 lines, you can use TensorFlow and TimeWarPY to create a fully trained model for time series forecasting. A comparison of the complexity decrease can be seen [here](https://www.tensorflow.org/tutorials/structured_data/time_series#data_windowing) in the TensorFlow docs.

```py
# load libraries
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from timewarpy import core, datasets

# load and preprocess
df = datasets.load_energy_data()
TSprocessor = core.UnivariateTS(1000, 100, scaler=MinMaxScaler)
X, y = TSprocessor.fit_transform(df, 'Appliances')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(f'Original dataframe shape: {df.shape}')
print(f'X training vector shape: {X_train.shape}')
print(f'y training vector shape: {y_train.shape}')

# train small tensorflow model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(10, activation='tanh', recurrent_activation='sigmoid', input_shape=X_train[0].shape))
model.add(tf.keras.layers.Dense(100))
model.compile(optimizer='Adam', loss='mean_squared_error',)
history = model.fit(X_train, y_train, epochs=2, batch_size=100,)

# make predictions
y_pred = model.predict(X_test)
mae = np.mean(np.abs(TSprocessor.inverse_transform(y_test - y_pred)))
print(f'y prediction vector shape: {y_pred.shape}')
print(f'Model Mean Absolute Error: {mae:.2f}')
```

```
Original dataframe shape: (19735, 29)
X training vector shape: (12486, 1000, 1)
y training vector shape: (12486, 100)
Epoch 1/2
125/125 [==============================] - 62s 463ms/step - loss: 0.0103
Epoch 2/2
125/125 [==============================] - 52s 416ms/step - loss: 0.0085
y prediction vector shape: (6150, 100)
Model Mean Absolute Error: 56.82
```

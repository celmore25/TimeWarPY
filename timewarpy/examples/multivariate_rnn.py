# load libraries
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from timewarpy import core, datasets

# load and preprocess
df = datasets.load_energy_data()
TSprocessor = core.MultivariateTS(1000, 100, scaler=MinMaxScaler)
X, y = TSprocessor.fit_transform(df, 'Appliances')
X, y = TSprocessor.fit_transform(df, train_columns=['Appliances', 'T1', 'RH_1'], pred_columns=['Appliances', 'T1'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(f'Original dataframe shape: {df.shape}')
print(f'X training vector shape: {X_train.shape}')
print(f'y training vector shape: {y_train.shape}')

# train small tensorflow model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(10, activation='tanh', recurrent_activation='sigmoid', input_shape=X_train[0].shape))
model.add(tf.keras.layers.Dense(200))
model.add(tf.keras.layers.Reshape((100, 2), input_shape=(200,)))
model.compile(optimizer='Adam', loss='mean_squared_error',)
history = model.fit(X_train, y_train, epochs=2, batch_size=100,)

# make predictions
y_pred = model.predict(X_test)
print(f'y prediction vector shape: {y_pred.shape}')

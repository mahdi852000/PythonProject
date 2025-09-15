import tensorflow as tf

tf.random.set_seed(42)

model = tf.keras.Sequential([tf.keras.layers.Dense(3, input_shape=(4,), activation='relu'),
                             tf.keras.layers.Dense(1)])
model.compile(optimizer='adam', loss='mse')
model.fit(tf.random.normal([32,4]), tf.random.normal([32,1]), epochs=1)
pred = model.predict(tf.random.normal([1,4]))
print(pred)

import tensorflow as tf

tf.random.set_seed(42)

X = tf.random.normal([256, 4])
y = tf.reduce_sum(X, axis=1, keepdims=True)  # هدف: جمع چهار ویژگی

model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_shape=(4,), activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X, y, epochs=20, batch_size=32, verbose=0)

test = tf.constant([[0.5, -1.0, 0.25, 0.25]])  # جمع=0.0
print("Pred:", model.predict(test))

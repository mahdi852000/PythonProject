import os
# اختیاری: کم‌کردن لاگ‌ها و یکسان‌تر کردن نتایج روی CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # اگر می‌خوای نتایج دقیقاً قابل‌تکرارتر باشن

import tensorflow as tf
tf.random.set_seed(42)

# داده: هدف = جمع چهار ویژگی
X = tf.random.normal([2000, 4])
y = tf.reduce_sum(X, axis=1, keepdims=True)

# مدل خطی: دقیقاً یک لایه Dense به‌صورت y ~= w1*x1 + ... + w4*x4
model = tf.keras.Sequential([
    tf.keras.Input(shape=(4,)),
    tf.keras.layers.Dense(1, activation=None, use_bias=False)
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss='mse', metrics=['mae'])

_ = model.fit(X, y, epochs=30, batch_size=64, verbose=0)

test = tf.constant([[0.5, -1.0, 0.25, 0.25]])  # جمع دقیق = 0.0
pred = model.predict(test, verbose=0)

pred_val = pred.item()
print("True sum:", float(tf.reduce_sum(test)))
# print("Pred:", pred, "Abs error:", abs(float(pred) - 0.0))
print(f"Pred: {pred_val:.6g}  Abs error: {abs(pred_val):.3e}")

t2 = tf.constant([[3.0, -1.0, 2.0, -4.0]])  # جمع = 0
print("Pred2:", model.predict(t2, verbose=0).item())

print("Weights ~", model.get_weights()[0].squeeze())


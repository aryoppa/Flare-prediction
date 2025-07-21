from tcn import TCN
import tensorflow as tf
model = tf.keras.models.load_model('trained_model/solarflare_model_plain.keras', custom_objects={'TCN': TCN})
print(model.summary())

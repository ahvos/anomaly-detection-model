import tensorflow as tf
import numpy as np
import joblib

# gen model
def build_generator(input_dim, feature_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_dim=input_dim),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(feature_dim, activation='sigmoid')
    ])
    return model


# main lstm model
def build_discriminator():
    detection_model = joblib.load(open('/Users/odagled/PythonProjs/ML4930/ML_Proj/anamoly_detect.ipynb', 'rb'))
    return detection_model


# noise
noise_dim = 100
feature_dim = 69

generator = build_generator(noise_dim, feature_dim)
#our_model = joblib.load(open('/Users/odagled/PythonProjs/ML4930/ML_Proj/anamoly_detect.ipynb', 'rb'))
discriminator = build_discriminator()

# train loop for adver
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
for epoch in range(15):
    noise = np.random.normal(0, 1, (32, noise_dim))
    generated_packets = generator.predict(noise)

    # check generated packets with main model
    detection_outputs = discriminator.predict(generated_packets)

    # loss
    loss = tf.reduce_mean(detection_outputs)

    # time for backprop


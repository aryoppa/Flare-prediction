# model/tcn_manual.py

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Dropout, Add, Dense, Flatten
import tensorflow.keras.backend as K


def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return K.sum(loss, axis=1)
    return loss

def residual_block(x, filters, dilation_rate):
    prev = x
    x = Conv1D(filters, kernel_size=3, padding='causal', dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    if prev.shape[-1] != filters:
        prev = Conv1D(filters, 1, padding='same')(prev)
    out = Add()([x, prev])
    return out

def build_manual_tcn_model(input_shape, num_classes):
    inp = Input(shape=input_shape)
    x = inp
    for dilation in [1, 2, 4, 8]:
        x = residual_block(x, filters=64, dilation_rate=dilation)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)
    return model

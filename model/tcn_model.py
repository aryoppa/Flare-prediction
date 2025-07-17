# model/train_model.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tcn import TCN


def build_model(input_shape, num_classes):
    input_layer = Input(shape=input_shape)
    x = TCN(nb_filters=64, kernel_size=3, dilations=[1, 2, 4, 8])(input_layer)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def run_training(X_train, y_train, X_val, y_val, class_weight=None):
    num_classes = y_train.shape[1]
    input_shape = X_train.shape[1:]

    model = build_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        class_weight=class_weight,
        callbacks=[reduce_lr],
        verbose=1
    )
    return model, history

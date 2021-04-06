import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf
from tensorflow.keras.layers import Input, BatchNormalization, GlobalAveragePooling2D, Dropout, Dense, LeakyReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, RandomFlip, RandomRotation, RandomZoom

IMAGE_SIZE = 64
train_dir = 'xrays-x64/train'
EPOCHS = 100
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

with tf.device('/GPU:0'):
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=(IMAGE_SIZE, IMAGE_SIZE)
    )

    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=(IMAGE_SIZE, IMAGE_SIZE)
    )

    augmentation = Sequential(
        [
            RandomFlip('horizontal', input_shape=IMG_SHAPE),
            RandomRotation((-0.1, 0.1)),
            RandomZoom(0.1),
        ]
    )

    vgg16_model = VGG16(input_shape=IMG_SHAPE,
                        include_top=False, weights='imagenet')
    vgg16_model.trainable = False

    model = Sequential([
        augmentation,
        Rescaling(1./255, input_shape=IMG_SHAPE),
        vgg16_model,
        BatchNormalization(name='BatchNormalization'),
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(256),
        LeakyReLU(alpha=0.1),
        Dropout(0.25),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10,
                                       verbose=1, mode='auto', min_delta=0.0001, cooldown=5, min_lr=0.0001)

    callbacks_list = [reduceLROnPlat]
    history = model.fit(train_dataset,
                        validation_data=validation_dataset,
                        epochs=EPOCHS,
                        callbacks=callbacks_list,
                        verbose=2
                        )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    model.save('vgg16-x64-new-checkpoint-100')

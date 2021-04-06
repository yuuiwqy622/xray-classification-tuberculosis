from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf
from tensorflow.keras.layers import Input, BatchNormalization, GlobalAveragePooling2D, Dropout, Dense, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

IMAGE_SIZE = 64
train_dir = 'xrays-x64/train'
EPOCHS = 100

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

    IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
    inputs = Input(IMG_SHAPE)
    CLASS_CNT = 2

    vgg16_model = VGG16(input_shape =  IMG_SHAPE, include_top = False, weights = 'imagenet')
    vgg16_model.trainable = False
    outputs = vgg16_model(inputs)
    outputs = BatchNormalization(name = 'BatchNormalization')(outputs)
    outputs = GlobalAveragePooling2D()(outputs)
    outputs = Dropout(0.5)(outputs)
    outputs = Dense(256)(outputs)
    outputs = LeakyReLU(alpha=0.1)(outputs)
    outputs = Dropout(0.25)(outputs)
    outputs = Dense(1, activation = 'sigmoid')(outputs)

    model = Model(inputs = [inputs], outputs = [outputs])
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()

    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=5, min_lr=0.0001)
    early = EarlyStopping(monitor="'val_acc'", mode="auto", patience=30) 

    callbacks_list = [early, reduceLROnPlat]
    history = model.fit(train_dataset,
             validation_data=validation_dataset,
             epochs=EPOCHS,
             callbacks=callbacks_list,
             verbose=2)
    model.save('vgg16-x64-new-checkpoint-100')

    # STEP_SIZE_TRAIN=train_gen.n // train_gen.batch_size
    # STEP_SIZE_VALID=valid_gen.n // valid_gen.batch_size

    # model.fit_generator(train_gen, steps_per_epoch=STEP_SIZE_TRAIN, validation_data=valid_gen, validation_steps=STEP_SIZE_VALID,
    #                     epochs = 100, callbacks = callbacks_list)
    # model.save('vgg16-x64-checkpoint-200')
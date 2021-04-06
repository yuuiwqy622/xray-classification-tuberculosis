from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf
from tensorflow.keras.layers import Input, BatchNormalization, GlobalAveragePooling2D, Dropout, Dense, LeakyReLU
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

IMAGE_SIZE = 64
TEST_DIR = 'xrays-x64/test'

with tf.device('/GPU:0'):
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_DIR,
        seed=123,
        image_size=(IMAGE_SIZE, IMAGE_SIZE)
    )

    model = load_model('vgg16-x64-new-checkpoint-100')
    # model = load_model('vgg16/vgg16-x64-checkpoint-100')
    results = model.evaluate(test_dataset)
    print(results)
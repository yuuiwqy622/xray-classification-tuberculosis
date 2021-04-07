import tensorflow as tf
import glob
from keras.preprocessing.image import load_img, img_to_array, save_img

print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))

DIR = 'xrays/train/0'
FILES = glob.glob(f'{DIR}/*.png')
TARGET_SIZE = (128, 128)

with tf.device('/GPU:0'):
    for f in FILES:
        img = load_img(f, target_size = TARGET_SIZE)
        img = img_to_array(img)
        save_path = f.split('/')
        save_path[0] = f'{save_path[0]}-x128'
        save_path = '/'.join(save_path)
        print(save_path)
        save_img(save_path, img)
import glob
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
import tensorflow as tf
from PIL import ImageOps
import numpy as np

directory = 'xrays-x64/test/1'
files = glob.glob(f'{directory}/*.png')

with tf.device('/GPU:0'):
    all_equal = True
    for f in files:
        img = load_img(f, target_size = (64, 64))
        img = img_to_array(img)
        all_equal = all_equal and np.array_equal(img[:,:,0], img[:,:,1]) 
        all_equal = all_equal and np.array_equal(img[:,:,0], img[:,:,2])
    print(all_equal)
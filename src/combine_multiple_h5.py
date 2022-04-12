import h5py
import numpy as np
from glob import glob
import tensorflow as tf
import os
from time import time

from parse_stanford3d import N_CATEGORIES, OUPUT_PATH



ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
DATA = os.path.join(ROOT, 'data')

files = os.path.join(DATA,'blocks_original_PN/train/*.h5')
OUPUT_PATH = os.path.join(DATA,'blocks_original_PN/joined')
N_CATEGORIES=13

print('-----initializing dataset-----')
dataset_data = np.zeros([0,4096,9],dtype = np.float32)
dataset_label = np.zeros([0,4096,13],dtype = np.float32)

for fn in glob(files):
    start = time()
    print('generator opens {}'.format(fn))
    with h5py.File(fn, 'r') as hf:
        for points, labels in zip(hf["data"],hf["label"]):
            labels = tf.keras.utils.to_categorical(labels,num_classes=13)
            points = np.expand_dims(points,0)
            labels = np.expand_dims(labels,0)

            dataset_data = np.append(dataset_data, points, axis=0)
            dataset_label = np.append(dataset_label, labels, axis=0)
    print('finished in {} sec'.format(time()-start))

np.save(os.path.join(OUPUT_PATH,'train_data'),dataset_data)
np.save(os.path.join(OUPUT_PATH,'train_label'),dataset_label)



import os
from glob import glob
#
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
#
from sklearn.model_selection import train_test_split
import h5py




class Generator_h5:
    def __init__(self, file):
        self.file = file
        #self.batch_size = batch_size
    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            
            for points, labels in zip(hf["points"],hf["labels"]):
                # labels = tf.keras.utils.to_categorical(labels,num_classes=13)
                # labels = np.zeros((4096,13))
                yield points, labels

class Generator_npy:
    def __init__(self, points, labels):
        self.points = np.load(points)
        self.labels = np.load(labels)
    def __call__(self):          
        for points, labels in zip(self.points, self.labels):
            labels = tf.keras.utils.to_categorical(labels, num_classes=13)
            yield points, labels

class Generator_multiple_h5:
    # to be called when training on original pointnet data
    def __init__(self, files):

        print('-----initializing dataset-----')
        self.dataset_data = np.zeros([0,4096,9],dtype = np.float32)
        self.dataset_label = np.zeros([0,4096,13],dtype = np.float32)
        for fn in glob(files):
          print('generator opens {}'.format(fn))
          with h5py.File(fn, 'r') as hf:
            for points, labels in zip(hf["data"],hf["label"]):
              labels = tf.keras.utils.to_categorical(labels,num_classes=13)
              points = np.expand_dims(points,0)
              labels = np.expand_dims(labels,0)
              self.dataset_data = np.append(self.dataset_data, points, axis=0)
              self.dataset_label = np.append(self.dataset_label, labels, axis=0)

    def __call__(self):            
      for points, labels in zip(self.dataset_data, self.dataset_label):
          yield points, labels



class Generator_multiple_h5_old:
    # to be called when training on original pointnet data
    def __init__(self, files):
        self.files = glob(files)
    def __call__(self):
        for file_at_n in self.files:
          print('generator opens {}'.format(file_at_n))
          with h5py.File(file_at_n, 'r') as hf:
              
              for points, labels in zip(hf["data"],hf["label"]):
                  labels = tf.keras.utils.to_categorical(labels,num_classes=13)
                  yield points, labels






if __name__ == '__main__': 

    pass



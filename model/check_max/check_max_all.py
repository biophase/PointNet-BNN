import numpy as np
import argparse 
from glob import glob
import os

# parser = argparse.ArgumentParser()
# parser.add_argument('file', type=str)
# args=parser.parse_args()

logs = glob('./*.txt')

for log in logs:
    values = np.loadtxt(log)
    print(os.path.split(log)[1])
    print('max value : {}'.format(np.max(values)))
    print('at index: {}'.format (np.argmax(values)))
    print('-'*30,end='\n\n')
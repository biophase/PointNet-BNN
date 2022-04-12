import numpy as np
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str)
args=parser.parse_args()

values = np.loadtxt(args.file)
print('max value : {}'.format(np.max(values)))
print('at index: {}'.format (np.argmax(values)))
import numpy as np
from glob import glob
import os
from time import time
import shutil
from plyfile import PlyData, PlyElement
import numpy as np


ROOT = os.path.split(os.path.abspath(__file__))[0]

OUTPUT_TRAIN = os.path.join(ROOT, 'train')
os.makedirs(OUTPUT_TRAIN, exist_ok=True)

OUTPUT_VAL = os.path.join(ROOT, 'val')
os.makedirs(OUTPUT_VAL, exist_ok=True)

INPUT_TRAIN = os.path.join(ROOT, '../8_2_Semi_synth_cs_arrange/train')
INPUT_VAL = os.path.join(ROOT, '../8_2_Semi_synth_cs_arrange/val')

train_ply_fps = glob(os.path.join(INPUT_TRAIN, '*.ply'))
print('Found {} training PLYs'.format(len(train_ply_fps)))

val_ply_fps = glob(os.path.join(INPUT_VAL, '*.ply'))
print('Found {} val PLYs'.format(len(val_ply_fps)))

def translate_ply_npy(input_fps, output_pth):
    for input_fp in input_fps:
        ply = PlyData.read(input_fp)
        ply = ply['vertex']
        ply = np.concatenate([
                            ply['x'][:,np.newaxis],
                            ply['y'][:,np.newaxis],
                            ply['z'][:,np.newaxis],
                            ply['red'][:,np.newaxis],
                            ply['green'][:,np.newaxis],
                            ply['blue'][:,np.newaxis], 
                            ply['scalar_label'][:,np.newaxis]
                            ],
                            axis = 1
        )
        
        # put leftmostcorner at 0,0,0
        ply[:,:3] = ply[:,:3] - np.min(ply[:,:3], axis = 0)
        new_fn = os.path.split(input_fp)[1].replace('.ply','.npy')
        print('saving: {}'.format(new_fn))
        np.save(os.path.join(output_pth, new_fn), ply)
        

print('Creating train NPY')        
translate_ply_npy(train_ply_fps, OUTPUT_TRAIN)
print('Creating val NPY')
translate_ply_npy(val_ply_fps, OUTPUT_VAL)
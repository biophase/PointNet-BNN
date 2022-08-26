import numpy as np
from glob import glob
import os
from time import time
import shutil
from plyfile import PlyData, PlyElement
import numpy as np


ROOT = os.path.split(os.path.abspath(__file__))[0]

input_dataset_name ='snrw_full_sr-0-40'
output_dataset_name='snrw_full_sr-0-40_12-classes'

OUTPUT_TRAIN = os.path.join(ROOT, f'1_npy/{output_dataset_name}/train')
os.makedirs(OUTPUT_TRAIN, exist_ok=True)

OUTPUT_VAL = os.path.join(ROOT, f'1_npy/{output_dataset_name}/val')
os.makedirs(OUTPUT_VAL, exist_ok=True)

INPUT_TRAIN = os.path.join(ROOT, f'0_ply/{input_dataset_name}/train')
INPUT_VAL = os.path.join(ROOT, f'0_ply/{input_dataset_name}/val')

print(ROOT)
train_ply_fps = glob(os.path.join(INPUT_TRAIN, '*.ply'))
print('Found {} training PLYs'.format(len(train_ply_fps)))

val_ply_fps = glob(os.path.join(INPUT_VAL, '*.ply'))
print('Found {} val PLYs'.format(len(val_ply_fps)))

# plyfile key (normally 'vertex' or 'pointcloud')
ply_key = 'pointcloud'

# list the names of fields to extract from the .ply
# the vector should strart with x,y,z,... and end with ground_truth
features=[
    'x',
    'y',
    'z',
    
    # 'red',
    # 'blue',
    # 'green'
    
    'intensity',
    
    # 'omnivariance',
    # 'eigenentropy',
    # 'anisotropy',
    
    'planarity',
    'linearity',
    'sphericity',
    'verticality',
    
    'nx',
    'ny',
    'nz',

    'norm_scene_x',
    'norm_scene_y',
    'norm_scene_z',

    'gt'
]

# ground truth map to merge classes
gt_map = {
    0 : 0,
    1 : 1,
    2 : 1,
    3 : 2,
    4 : 2,
    5 : 3,
    6 : 1,
    7 : 1,
    8 : 4,
    9 : 4,
    10 : 5,
    11 : 5,
    12 : 6,
    13 : 1,
    14 : 7,
    15 : 8,
    16 : 9,
    17 : 10,
    18 : 11

}


def translate_ply_npy(input_pths, output_pth, features=features):
    for input_pth in input_pths:
        ply = PlyData.read(input_pth)
        ply = ply[ply_key]
        ply ['gt'] = np.vectorize(gt_map.get)(ply['gt'])
        ply = np.concatenate([ply[feature][:,np.newaxis] for feature in features],axis = 1)
        ply = np.nan_to_num(ply)
        
        # put leftmostcorner at 0,0,0
        ply[:,:3] = ply[:,:3] - np.min(ply[:,:3], axis = 0)

        # duplicate coordinates

        # normalize features 
        ply = np.nan_to_num(ply, nan=0.)
        ply[:,3:-1] = (ply[:,3:-1] - np.min(ply[:,3:-1],axis=0))
        ply[:,3:-1] /= np.max(ply[:,3:-1],axis = 0)

        # save .npy
        new_fn = os.path.split(input_pth)[1].replace('.ply','.npy')
        print('saving: {}'.format(new_fn))
        np.save(os.path.join(output_pth, new_fn), ply)
        

print('Creating train NPY')        
translate_ply_npy(train_ply_fps, OUTPUT_TRAIN)
print('Creating val NPY')
translate_ply_npy(val_ply_fps, OUTPUT_VAL)
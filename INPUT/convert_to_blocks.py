import os
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
import numpy as np
#
import os
from glob import glob
import time
import parse_pointcloud
from sklearn.model_selection import train_test_split
import h5py
from pathlib import Path

from viz_utils import bcolors

ROOT = os.path.split(os.path.abspath(__file__))[0]
INPUT_DATASET_NAME = 'david_grid_ss'
OUTPUT_DATASET_NAME = 'david_grid_ss_BS-20-00'
INPUT_PATH = os.path.join(ROOT,f'1_npy/{INPUT_DATASET_NAME}/')
OUPUT_PATH = os.path.join(ROOT,f'2_h5/{OUTPUT_DATASET_NAME}/')

# Parameters
N_CATEGORIES = 12
BLOCK_SIZE = [20.,20.,100.]


##################
## Strassen NRW ##
##################

# CATEGORIES=[
#     'ifcbridge_abutment',
#     'ifcbridge_beam',
#     'ifcbridge_girder',
#     'ifcbridge_pile',
#     'ifcbridge_pier',
#     'ifcbridge_pier_footing',
#     'ifcbridge_railing',
#     'ifcbridge_span',
#     'ifcearthworkfill',
#     'ifcplant',
#     'ifcroad',
#     'ifcroadside',
#     'ifcroad_railing',
#     'ifcslab',
#     'ifctransportelement', # -> rest
#     'noise_barrier',
#     'sign_bridge',
#     'traffic_sign',    
#     'rest'
# ]

###############
## Scaf_sing ##
###############

# COLOR_MAP= { # ScafSing
#     0: np.array((72,72,204)),
#     1: np.array((162, 209, 125)),
#     2: np.array((204,72,72))
# }

# CATEGORIES={
#     "construction site":0, 
#     "building":1, 
#     "scaffold":2,
#     "tree":3
# }
# COLOR_MAP= { # ScafSemi
#     0: np.array((72,72,204)),
#     1: np.array((162, 209, 125)),
#     2: np.array((204,72,72)),
#     3: np.array((72,72,72))
# }


###########
## S3DIS ##
###########

# CATEGORIES ={
#     'ceiling'     :   0,
#     'floor'       :   1,
#     'wall'        :   2,
#     'beam'        :   3,
#     'column'      :   4,
#     'window'      :   5,
#     'door'        :   6,
#     'table'       :   7,
#     'chair'       :   8,
#     'sofa'        :   9,
#     'bookcase'    :   10,
#     'board'       :   11,
#     'clutter'     :   12
# }

# COLOR_MAP = {
#     0           :np.array((97,47,0)),     # dark brown      --> ceiling
#     1           :np.array((219,34,210)),  # violet          --> floor
#     2           :np.array((168,100,35)),  # light brown     --> wall
#     3           :np.array((64,30,219)),   # blue            --> beam
#     4           :np.array((219,13,0)),    # red             --> column
#     5           :np.array((106,219,75)),  # green           --> window
#     6           :np.array((0,237,211)),   # cyan            --> door
#     7           :np.array((2,156,81)),    # dark green      --> table
#     8           :np.array((217,213,0)),   # yellow          --> chair
#     9           :np.array((98,0,168)),    # dark purple     --> sofa
#     10          :np.array((255,103,145)), # pink            --> bookcase
#     11          :np.array((138,62,57)),   # dark pink       --> board
#     12          :np.array((130,130,130)), # gray            --> clutter

# }


#########
# ARCH ##
#########

# CATEGORIES={
#     "arch":0, 
#     "column":1, 
#     "moldings":2, 
#     "floor":3, 
#     "door_window":4, 
#     "wall":5, 
#     "stairs":6, 
#     "vault":7, 
#     "roof":8, 
#     "other":9}
# COLOR_MAP = {
#     0           :np.array((97,47,0)),     # dark brown      --> ceiling
#     1           :np.array((219,34,210)),  # violet          --> floor
#     2           :np.array((168,100,35)),  # light brown     --> wall
#     3           :np.array((64,30,219)),   # blue            --> beam
#     4           :np.array((219,13,0)),    # red             --> column
#     5           :np.array((106,219,75)),  # green           --> window
#     6           :np.array((0,237,211)),   # cyan            --> door
#     7           :np.array((2,156,81)),    # dark green      --> table
#     8           :np.array((217,213,0)),   # yellow          --> chair
#     9           :np.array((98,0,168)),    # dark purple     --> sofa
# }
    

###



# def parse_stanford_3d():
#     '''Use this to parse the original S3DIS, where the annotations of each room are themselves
#         separate .txt files. The output are temporary .npy files containing the class as a label
#         appended at the end.'''
#     # loop through areas
#     for area in os.listdir(DATA):
#         if area.startswith('Area'):
#             area_path = os.path.join (DATA,area)
#             print('*'*20, 'Area {}'.format(area),'*'*20)
#             # loop through rooms  
#             for room in [room for room in os.listdir(area_path) if os.path.isdir(os.path.join(area_path,room))]:
                
#                 room_path           = os.path.join(area_path, room)
#                 annotations_path    = os.path.join(room_path,'Annotations')
#                 room_pointcloud     = np.empty([0,7],dtype=np.float32)
#                 n_files_in_dir = len(os.listdir(annotations_path))
                

#                 print('Parsing room ({}) with {} annotations'.format(room, n_files_in_dir))
                
                
#                 start = time.time()
                
#                 #loop through annotations
#                 for i,annotation in enumerate(glob(os.path.join(annotations_path,'*.txt'))):
                    
#                     category = CATEGORIES[os.path.split(annotation)[1].split('_')[0]]

#                     with open(annotation,'r') as annotation_file:
#                         start_ann = time.time()
#                         annot_size = os.path.getsize(annotation)/1000
#                         print("Parsing Annotation {}/{} ({:0.2f} KB)".format(i, n_files_in_dir, annot_size),end="\r")
#                         points_at_category = np.genfromtxt(annotation_file, delimiter = ' ')
#                         labels = np.full((points_at_category.shape[0],1),category)
#                         points_at_category_with_labels = np.append(points_at_category, labels, axis=-1)
#                         print(' '*55,'done in: {:0.2f} seconds'.format(time.time()-start_ann),end="\r")
#                         room_pointcloud = np.append(room_pointcloud, points_at_category_with_labels, axis = 0)
                        
                        

#                 print('Room {} took: {:0.2f}'.format(room, time.time()-start))

#                 # rooms.append(room)
#                 area_path_npy = os.path.join(DATA,'s3dis_npy',area)
#                 Path(area_path_npy).mkdir(parents=True, exist_ok=True)
#                 room_path_npy = os.path.join(area_path_npy,room + '.npy')
#                 print('Saving to {}'.format(room_path_npy))    
#                 np.save(room_path_npy, room_pointcloud)
#     # print(rooms.__len__())


# def parse_stanford_3d_blocks():
    
#     # loop through areas
#     for area in os.listdir(INPUT_PATH):
#         area_path = os.path.join (INPUT_PATH,area)
#         # loop through rooms  
#         for room in glob(os.path.join(area_path,'*.npy')):
#             start = time.time()
#             print('Parsing file ({})'.format(os.path.split(room)[1]))
            
#             points_colors_labels__room = np.load(room)
#             blocks,_= parse_pointcloud.get_2d_voxels(points_colors_labels__room[::1,...])
#             #print('Saving to {}'.format(os.path.join(OUPUT_PATH, os.path.split(room)[1]))) 
#             print('\n',end="")
#             print('Saving...', end ="\t")
#             os.makedirs(OUPUT_PATH, exist_ok=True)
#             np.save(os.path.join(OUPUT_PATH, os.path.split(room)[1]), blocks)
#             print('Done in {:0.2f}'.format(time.time() - start ), end = '\n\n')

        


def split(pcd_paths, points_dataset, labels_dataset, b_size, n_categories = N_CATEGORIES):
    
    for pcd_path in pcd_paths:
        start = time.time()
        
        include_stride = True

        print('Parsing file ({})'.format(os.path.split(pcd_path)[1]))
        

        stride =   [b_size[0]/2,
                    b_size[1]/2,
                    0]

        

        points_colors_labels__room = np.load(pcd_path) 
        points_colors_labels__room[:,:3] = points_colors_labels__room[:,:3] - np.min(points_colors_labels__room[:,:3], axis = 0)
        blocks,_,__= parse_pointcloud.get_3d_voxels(points_colors_labels__room,voxel_size=b_size)
        if include_stride:
            axes = ['x','y','z']
            print(f"Applying stride of {['{}: {:.2f}'.format(ax, dim) for ax,dim in zip(axes,stride)]}")
            points_colors_labels__room[:,:3] += stride
            blocks_strided,_,__= parse_pointcloud.get_3d_voxels(points_colors_labels__room,voxel_size=b_size)
            blocks = np.append(blocks, blocks_strided, axis = 0)

        print('\n',end="")
        print('Appending...', end ="\t")
        
        
        # increase dataset size by number of parsed chunks
        n_parsed_chunks = blocks.shape[0]
        current_last_chunk_in_ds = points_dataset.shape[0]
        points_dataset.resize(current_last_chunk_in_ds + n_parsed_chunks, axis=0)
        labels_dataset.resize(current_last_chunk_in_ds + n_parsed_chunks, axis=0)
        
        # append the parsed chunks/blocks
        points_dataset[-n_parsed_chunks: , ...] = blocks [:,:,:-1]
        
        # convert label to one-hot vector and append
        
        labels = tf.keras.utils.to_categorical(blocks[:,:,-1], num_classes = n_categories)
        
        labels_dataset[-n_parsed_chunks: , ...] = labels

        print('Done in {:0.2f}'.format(time.time() - start ), end = '\n\n')


def split_npy_in_blocks_h5(b_size):
    
    os.makedirs(OUPUT_PATH, exist_ok = True)

    
    
    # get data paths
    train_rooms = np.array(glob(os.path.join(INPUT_PATH,'train/*.npy')))
    val_rooms = np.array(glob(os.path.join(INPUT_PATH,'val/*.npy')))
    num_feats = np.load(train_rooms[0]).shape[-1] - 1 # get num of features and (-1) for labels

    # initialize h5 files
    train_data =  h5py.File(os.path.join(OUPUT_PATH,'train.h5'),mode = 'w')
    train_points_ds = train_data.create_dataset('points',shape=(0,4096,num_feats),maxshape=(None,4096,num_feats))
    train_labels_ds = train_data.create_dataset('labels',shape=(0,4096,N_CATEGORIES),maxshape=(None, 4096, N_CATEGORIES))

    val_data =  h5py.File(os.path.join(OUPUT_PATH,'val.h5'),mode = 'w') 
    val_points_ds = val_data.create_dataset('points',shape=(0,4096,num_feats),maxshape=(None,4096,num_feats))
    val_labels_ds = val_data.create_dataset('labels',shape=(0,4096,N_CATEGORIES),maxshape=(None, 4096, N_CATEGORIES))        
    if len(train_rooms) == 0 : raise Exception('Pointclouds must be in a "train" and "val folders"')
    print('Found {} point clouds for testing set\nFound {} point clouds for validation set\n'\
        .format(len(train_rooms),len(val_rooms)))
        
    start = time.time()
    split(train_rooms, train_points_ds, train_labels_ds, b_size=b_size)
    split(val_rooms, val_points_ds, val_labels_ds, b_size=b_size)        
    
    # print results
    print('Finished! Created:\
        \n\t- Training set with shapes:\
        \n\t\t- {} points,\
        \n\t\t- {} labels\
        \n\t- Validation set with shapes: \
        \n\t\t- {} points,\
        \n\t\t- {} labels\
        \nTotal time: {:0.2f}'\
        .format(train_points_ds.shape, train_labels_ds.shape, val_points_ds.shape, \
            val_labels_ds.shape, time.time()-start))






if __name__ == '__main__': 
    # parse_stanford_3d()
    split_npy_in_blocks_h5(b_size = BLOCK_SIZE)



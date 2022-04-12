from ast import Pass
import os

from tensorflow.python.ops.gen_math_ops import ceil, floor
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import numpy as np
import open3d as o3d
import tensorflow as tf
import model as model
import time
from glob import glob
import math
from plyfile import PlyData, PlyElement

import parse_pointcloud
from parse_stanford3d import COLOR_MAP
from visualize import bcolors

# load samples and model
samples_starting_with = '' # for a single file just put the name without .npy
model_name = '20220216_1959' 
# note : best model so far: 20211216_1852, 
# best bayesian model so far : 20211223_0008


MODEL_TYPES = {
    'bayesian'   : model.get_model_segmentation_bayesian,
    '1024'       : model.get_model_segmentation
}

# viz settings
show_true_labels = False #gt
show_ceiling = True
render_origin = False
show_stddev = True
show_boundaries = True

# model settings
model_type = 'bayesian' # options: 'bayesian', '64', '1024'
batch_size = 24

# output
save_pred = True # set to False to view result or True to save to .txt

# setup paths
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
# ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
samples_path_prefix = os.path.join(ROOT,'data/scaf_sing/test/' + samples_starting_with)

# find optimal checkpoint
model_path = os.path.join(ROOT,'model/checkpoints',model_name)
val_miou = np.loadtxt(os.path.join(model_path, 'val_miou_container.txt'))
best_checkpoint_indx = np.argmax(val_miou)
checkpoints = glob(os.path.join(model_path, '*.data-00000-of-00001'))
checkpoint = os.path.split(checkpoints[best_checkpoint_indx])[1].split('.')[0]
model_path = os.path.join(model_path, checkpoint)

# load model
print("\nLoading segmentation model...")
print("Using checkpoint {} with mIoU {}{:0.4f}{}".format(checkpoint, bcolors.OKGREEN, \
                                                val_miou[best_checkpoint_indx], bcolors.ENDC ), end ='\n')
bn_momentum = tf.Variable(.99, trainable=False)
inference_model = MODEL_TYPES[model_type](bn_momentum=bn_momentum, n_classes=COLOR_MAP.__len__())
load_status = inference_model.load_weights(model_path)
load_status.assert_consumed()
print('Done!')



#################
## Definitions ##
#################
def get_u_pred(preds):
    average_preds = np.average(preds, axis = -1)
    u_pred = -np.sum(average_preds*np.log2(average_preds),axis=-1)
    
    # return np.interp(u_pred,(u_pred.min(),u_pred.max()), (0,1))
    return u_pred

def get_u_alea(preds):
    entropy = np.sum(preds*np.log2(preds),axis =-2)
    entropy = -np.sum(entropy, axis = -1)/preds.shape[-1]
    return entropy




#################
## Load sample ##
#################

sample_paths = glob(samples_path_prefix + '*.npy')

if save_pred: output_path = os.path.join(ROOT,'data','test_pred','pred_output_'+str(int(time.time())))
for i, test_sample_path in enumerate(sample_paths):
    
    sample_name = os.path.split(test_sample_path)[1]

    # load pointcloud
    print("\nLoading File {} [{}/{}]".format(sample_name,i,len(sample_paths)))
    test_sample = np.load(test_sample_path)#.T ##################################################3
    test_sample = test_sample*[1,1,1,1/255.,1/255.,1/255.,1]
    test_sample = parse_pointcloud.pcd_corner_to_zero (test_sample)
###################    

    apply_voting_stride = True
    if apply_voting_stride:
      voting_stride = [0.5,0.5,0.0]  
      test_sample[:,:3] = test_sample[:,:3] + voting_stride
    # get blocks  
    test_sample_blocks,voxel_origins,_ = parse_pointcloud.get_2d_voxels(test_sample)

####################

    
    if save_pred: 
        points_labels_whole_file = np.zeros([0,12],dtype=np.float64)
        gt_labels_whole_file = np.zeros([0,1],dtype=np.float64)
    else:
        points_labels_whole_file = np.zeros([0,6],dtype=np.float64)
    n_batches = math.ceil(test_sample_blocks.shape[0]/batch_size)
    print ("{}Performing inference with {} batches{}".format(bcolors.OKGREEN,n_batches,bcolors.ENDC))

    voxel_boundaries = []
    for batch in range(n_batches):
        test_sample_block = test_sample_blocks[batch*batch_size : (batch+1)*batch_size , ...]
        test_sample_origins = voxel_origins[batch*batch_size : (batch+1)*batch_size]
        print('{}Batch {} with shape {}{}'.format(bcolors.OKCYAN, batch, test_sample_block.shape, bcolors.ENDC))

        # do inference
        label_true = test_sample_block[...,-1]
        points = test_sample_block[..., :9]
        # points = test_sample_block[..., :6]

        if model_type.startswith('bayesian'):
            
            # bayesian model inference
            cycles = 50
            print('Starting bayesian inference with {} cycles'.format(cycles))
            start = time.time()
            pred = np.zeros(inference_model(points,training=False).shape)
            preds = np.zeros((*pred.shape, 0))
            # stddev = np.expand_dims(pred, axis = -1)

            for i in range(cycles):
                cycle_pred = inference_model(points,training=False)
                print('[','+'*math.ceil(i/(cycles/30)),'-'*math.floor((cycles-i)/(cycles/30)),']',sep='', end ='\r')
                cycle_pred = np.expand_dims(cycle_pred,axis = -1)
                preds = np.append(preds, cycle_pred, axis=-1)
            pred = np.sum(preds,axis=-1)/preds.shape[-1]

            print('\n### Time bayesian inference took was: {:0.03f}'.format(time.time()-start))
            print('preds shape is: ', preds.shape)
            print('pred shape is: ', pred.shape)
            
           

            
            
            
            ###
            stddev = np.std(preds, axis = -1, keepdims=False)
            # print('stddev shape is: ', stddev.shape)
            u_pred = get_u_pred(preds)
            u_alea = get_u_alea(preds)
            u_epis= u_pred-u_alea
            # print('u_pred shape is: ', u_pred.shape)
            print('u_pred range is : {} to {}'.format(u_pred.min(),u_pred.max()))
            print('u_alea range is : {} to {}'.format(u_alea.min(),u_alea.max()))
            print('u_epis range is : {} to {}'.format(u_epis.min(),u_epis.max()))

            ####
            # stddev_per_point = np.mean(stddev, axis = -1, keepdims=True)
            value_per_point = u_epis[...,np.newaxis] # set value to display in viewport
            value_per_point = np.interp(value_per_point,(value_per_point.min(),value_per_point.max()), (0,1))

            
            # stddev_per_point_norm = value_per_point/np.max(value_per_point,axis=-2,keepdims=True) # normalize stddev
            
            stddev_per_point_color = np.append(value_per_point,value_per_point,axis=-1) # append stddev
            stddev_per_point_color = np.append(stddev_per_point_color,value_per_point,axis=-1) #
            color_multiplier = np.array([[[255,0,0]]])
            stddev_per_point_color = stddev_per_point_color*color_multiplier
            
            # print('heres one color:' ,np.mean(stddev_per_point_color, axis = 1))
        else:
            # normal inference
            pred = inference_model(points,training=False)
            

        # viewport gt?
        if show_true_labels: labels_to_show = tf.keras.utils.to_categorical(label_true, 14)
        else: labels_to_show = pred
        
        # if viewport show scalar or rgb?
        if show_stddev: nn_prediction = stddev_per_point_color

        # save ?
        if not save_pred : nn_prediction = parse_pointcloud.get_color_from_pred(labels_to_show, COLOR_MAP)
        else: 
            
            
            nn_prediction 
            nn_prediction = np.argmax(labels_to_show,axis=-1)[...,np.newaxis]
            
        
        



        # relocate points to original location
        rgb = points[...,3:6]*255
        points = points[...,:3]
        restuructured_points = np.zeros(points.shape)
        
        for i,block in enumerate(points):
            translation_per_point = np.zeros(block.shape)
            translation_per_point = np.tile(test_sample_origins[i],[4096,1])
            restuructured_points[i,...] = block + translation_per_point    
            voxel_boundaries.append(parse_pointcloud.draw_pcd_bbox(restuructured_points[i,...]))

        # print('mean values per block:\n' ,np.mean(labels, axis = 1)/255.0)
        # print('color shape is: ................', labels.shape)
        # print('points shape: ..................',points.shape)
        # print('origins shape: .................', voxel_origins.__len__())
        # print('restuructured_points shape is:..', restuructured_points.shape)
        # print('labels shape is: ...............', labels.shape)

        
        if save_pred: 
            # pred vector:
            # [x, y, z, r, g, b, pred, gt, u_pred, u_alea, u_epis]
            # print(restuructured_points.shape)
            # print(rgb.shape)
            # print(nn_prediction.shape)
            
            label_true = label_true[...,np.newaxis]
            # print(label_true.shape)
            
            u_pred = u_pred[...,np.newaxis]
            # print(u_pred.shape)
            u_alea = u_alea[...,np.newaxis]
            # print(u_alea.shape)
            u_epis = u_epis[...,np.newaxis]
            # print(u_epis.shape)

            error = np.where(nn_prediction == label_true, 0,1)
            # print(error.shape)
            
         
            
            points_labels = np.concatenate((restuructured_points,rgb , nn_prediction, label_true, u_pred, u_alea, u_epis, error),axis = -1)
            # print(points_labels.shape)

            points_labels = points_labels.reshape((-1,12))
            # label_true = label_true.reshape((-1,1))


        
        
        else: 
            points_labels = np.concatenate((restuructured_points,nn_prediction),axis = -1)
            points_labels = points_labels.reshape((-1,6))
    
    
        # print('point_labels.shape = ', points_labels.shape)
        points_labels_whole_file = np.append(points_labels_whole_file, points_labels, axis = 0)   
        print('points_labels_whole_file.shape = ', points_labels_whole_file.shape)
    
    if save_pred :
        # gt_labels_whole_file = np.append(gt_labels_whole_file, label_true,axis=0)

        if apply_voting_stride: 
          points_labels_whole_file[:,:3] = points_labels_whole_file[:,:3] - voting_stride
          
        
        os.makedirs(output_path, exist_ok=True)
        print('Saving file {}'.format(sample_name))
        # np.savetxt(os.path.join(os.path.abspath(output_path), sample_name.split('.')[0]+'_pred.txt'), points_labels_whole_file)
        print('Converting to PLY ...', end="")
        start_ply = time.time()
        
        
        points_labels_whole_file = list(map(tuple,points_labels_whole_file))
        # [x, y, z, r, g, b, pred, gt, u_pred, u_alea, u_epis, error]
        dtype = np.dtype([
                        ('x','f4'),
                        ('y','f4'),
                        ('z','f4'),
                        ('red','i4'),
                        ('green','i4'),
                        ('blue','i4'),
                        ('pred','f4'),
                        ('gt','i4'),
                        ('u_pred','f4'),
                        ('u_alea','f4'),
                        ('u_epis','f4'),
                        ('error','i4')
                        ])

        ply = np.array(points_labels_whole_file, dtype = dtype)
        el = PlyElement.describe(ply, 'pointcloud')
        PlyData([el]).write(os.path.join(os.path.abspath(output_path), sample_name.replace('.npy','_labeled.ply')))
        print('Done! , Took: {:.2f}'.format(time.time()-start_ply))
        tf.keras.backend.clear_session()

# crop ceiling

crop_height = 2.0
if not show_ceiling:
    points_labels_whole_file = points_labels_whole_file[np.where(points_labels_whole_file[:,2]<= crop_height)]

# print('points_labels_whole_file.shape = ', points_labels_whole_file.shape)
# restuructured_points = points_labels_whole_file[:,:3]
# nn_prediction = points_labels_whole_file[:,3:]/255.

# vizualize
# print('labels_shape: ', labels.shape)
# print('r_points shape: ',restuructured_points.shape)

if not save_pred:
    pointcloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(restuructured_points))

    pointcloud.colors = o3d.utility.Vector3dVector(nn_prediction)

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0,origin = [0.,0.,0.])
    if show_boundaries: render_items = [pointcloud,*voxel_boundaries]
    else: render_items = [pointcloud]


    if render_origin:
        render_items.append(origin)
    viz = o3d.visualization.draw_geometries(render_items, mesh_show_wireframe = True)
    print(viz)



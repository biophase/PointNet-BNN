from plyfile import PlyData, PlyElement
from datetime import timezone, datetime
from scipy import spatial
import numpy as np
from glob import glob
import os
from time import time

class WorkflowMode():
    FOR_VIZ = 1
    FOR_EVAL = 2
def get_timestamp():
    timestamp = str(datetime.now(timezone.utc))[:16]
    timestamp = timestamp.replace('-', '')
    timestamp = timestamp.replace(' ', '_')
    timestamp = timestamp.replace(':', '')
    return timestamp
def ply_to_npy(arr):
    shape_like = arr.elements[0].data['x'][:,np.newaxis]
    npy_arr = np.concatenate([
                arr.elements[0].data['x'][:,np.newaxis],
                arr.elements[0].data['y'][:,np.newaxis],
                arr.elements[0].data['z'][:,np.newaxis],

                arr.elements[0].data['red'][:,np.newaxis],
                arr.elements[0].data['green'][:,np.newaxis],
                arr.elements[0].data['blue'][:,np.newaxis],

                arr.elements[0].data['pred'][:,np.newaxis],
                arr.elements[0].data['gt'][:,np.newaxis],

                arr.elements[0].data['u_pred'][:,np.newaxis],
                arr.elements[0].data['u_alea'][:,np.newaxis],
                arr.elements[0].data['u_epis'][:,np.newaxis],

                arr.elements[0].data['error'][:,np.newaxis],
                # np.ones_like(shape_like, dtype = float)*original_cloud_indx
                ],axis=-1)
    return npy_arr

dataset_name ='snrw_full_sr-0-40_12-classes'

INIT_TIMESTAMP = get_timestamp()
WORKFLOW = WorkflowMode.FOR_VIZ # select mode
ROOT = os.path.split(os.path.abspath(__file__))[0]
INPUT = os.path.join(ROOT, 'input',dataset_name)
OUTPUT = os.path.join(ROOT,'output',INIT_TIMESTAMP)



stride_fps = glob(os.path.join(INPUT,'*_stride.ply'))
nostride_fps = [fp.replace('_stride.ply','_nostride.ply') for fp in stride_fps]


time_deltas = []
ns_points = []
for stride_fp, nostride_fp in zip(stride_fps, nostride_fps):
    
    output_pcd_fname = os.path.split(stride_fp)[1].replace('stride','corrected')
    print('Generating file: ',output_pcd_fname)
    nostride_ply = PlyData.read(nostride_fp)
    stride_ply = PlyData.read(stride_fp)


    nostride_pcd = ply_to_npy(nostride_ply)
    # print('no-stride shape: ', nostride_pcd.shape)
    stride_pcd = ply_to_npy(stride_ply)
    # print('stride shape: ',stride_pcd.shape)



    start = time()

    
    kd_nostride = spatial.KDTree(nostride_pcd[:,:3])
    kd_stride = spatial.KDTree(stride_pcd[:,:3])

    indexes = kd_nostride.query_ball_tree(kd_stride, r = 0.1)
    n_points = len(indexes)
    print('Number of points in non-strided pointcloud : ',n_points)

    def correct_for_viz():
        corrected_pcd = np.zeros(shape = [nostride_pcd.shape[0],13], dtype= float )
        # print(corrected_pcd.shape)
        for i in range(len(indexes)): # loop through nostride points
            for j in indexes[i]: # loop through points in stride within 0.1 of ns_point[i]
                ns_point = nostride_pcd [i,:]#[np.newaxis, :]
                s_point = stride_pcd [j,:]#[np.newaxis, :]
                # pair = np.concatenate([ns_point, s_point])
                # print(np.argmax(pair[:,8]))
                correction_needed = ns_point[8] < s_point[8]
                corrected_point = np.zeros(13)
                if correction_needed:
                    corrected_point[:12] = ns_point
                else:
                    corrected_point[:12] = ns_point
                    corrected_point[3:12] = s_point[3:]
                
                corrected_point[12] = correction_needed
                corrected_pcd[i,:] = corrected_point
                
        # print(corrected_pcd.shape)

        time_delta = time()-start
        print('Correction Done! Took : {:.2f}'.format(time_delta ))   
            
        corrected_pcd = list(map(tuple,corrected_pcd))
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
                        ('error','i4'),
                        ('correction','i4')
                        ])
        return corrected_pcd, dtype, time_delta
    def correct_for_eval():
        corrected_pcd = np.zeros(shape = nostride_pcd.shape, dtype= float )
        for i in range(len(indexes)): # loop through nostride points
            for j in indexes[i]: # loop through points in stride within 0.1 of ns_point[i]
                ns_point = nostride_pcd [i,:]#[np.newaxis, :]
                s_point = stride_pcd [j,:]#[np.newaxis, :]
                # pair = np.concatenate([ns_point, s_point])
                # print(np.argmax(pair[:,8]))
                if ns_point[8] < s_point[8]:
                    corrected_point = ns_point
                else:
                    corrected_point = s_point
                corrected_pcd[i,:] = corrected_point
        # print(corrected_pcd.shape)
        time_delta = time()-start
        print('Correction Done! Took : {:.2f}'.format(time_delta ))   
            
        corrected_pcd = list(map(tuple,corrected_pcd))
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
                        ('error','i4'),
                    ])      
        return corrected_pcd, dtype, time_delta

    if WORKFLOW == WorkflowMode.FOR_EVAL: correction_fn = correct_for_eval
    if WORKFLOW == WorkflowMode.FOR_VIZ: correction_fn = correct_for_viz

    corrected_pcd, dtype, time_delta = correction_fn()
    
    
    start_ply = time()
    ply = np.array(corrected_pcd, dtype = dtype)
    el = PlyElement.describe(ply, 'pointcloud')
    os.makedirs(OUTPUT, exist_ok='True')
    PlyData([el]).write(os.path.join(OUTPUT, output_pcd_fname))
    print('Done!, Took: {:.2f}'.format(time()-start_ply))

    time_deltas.append(time_delta)
    ns_points.append(n_points)

    np.savetxt(os.path.join(OUTPUT, 'time_deltas'),time_deltas)
    np.savetxt(os.path.join(OUTPUT, 'ns_points'),ns_points) 

    

    
    
    





# ply = PlyData.read('')
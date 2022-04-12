from plyfile import PlyData, PlyElement
from datetime import timezone, datetime
import numpy as np
from glob import glob
import os
from time import time
import sys
# import tensorflow as tf
from sklearn import metrics
from matplotlib import pyplot as plt
import matplotlib.font_manager as font_manager

sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))
from visualize import bcolors


class NumCategories():
    s3dis = 13
    arch = 10
    scaf_sing = 3
    scaf_semi = 4
    scaf_real = 4

def get_timestamp():
    timestamp = str(datetime.now(timezone.utc))[:16]
    timestamp = timestamp.replace('-', '')
    timestamp = timestamp.replace(' ', '_')
    timestamp = timestamp.replace(':', '')
    return timestamp
def ply_to_npy(arr, correction = False):
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
    if correction: np.concatenate(npy_arr, arr.elements[0].data['error'][:,np.newaxis], axis =-1)
    return npy_arr

def ply_get_field(ply_el, field_name):
    return ply_el.elements[0].data[field_name]

def rescale_array(arr, down, up):
    arr = np.interp(arr, (arr.min(), arr.max()), (down, up))
    return arr
#####
# ------->
CORRECTION_TIMESTAMP = '20220216_1936_combi_correctstride'
#####

INIT_TIMESTAMP = get_timestamp()
ROOT = os.path.split(os.path.abspath(__file__))[0]
INPUT_PREDICTED = os.path.join(ROOT, 'scaf_combi')
INPUT_CORRECTED = os.path.join(ROOT,'output',CORRECTION_TIMESTAMP)
FONT = {'fontname':'Times New Roman'}
FONT_LEGEND = font_manager.FontProperties(family='Times New Roman',style='normal', size=9)

NUM_CATEGORIES = NumCategories.scaf_sing


stride_fps = glob(os.path.join(INPUT_PREDICTED,'*_stride.ply'))
nostride_fps = [fp.replace('_stride.ply','_nostride.ply') for fp in stride_fps]
corrected_fps = [os.path.join(INPUT_CORRECTED, os.path.split(fp.replace('_stride.ply', '_corrected.ply'))[1]) for fp in stride_fps]

# Containers
time_deltas = []
ns_points = []
cal_crv_plotted = False
cal_crv = []
obs_probs_all = []
x_s_graph = []
ious = []
# Loop over files
for stride_fp, nostride_fp, corrected_fp in zip(stride_fps, nostride_fps, corrected_fps):
    
    # Load data
    fname = os.path.split(stride_fp)[1]
    room_name = fname.split('_')
    room_name = '_'.join(room_name[:4])
    
    
    nostride_ply = PlyData.read(nostride_fp)
    stride_ply = PlyData.read(stride_fp)
    corrected_ply = PlyData.read(corrected_fp)

    nostride_acc = 1 - np.average(ply_get_field(nostride_ply,'error'))
    stride_acc = 1 - np.average(ply_get_field(stride_ply,'error'))
    corrected_acc = 1 - np.average(ply_get_field(corrected_ply,'error'))

    improvement = corrected_acc - np.mean([nostride_acc,stride_acc])

#####################################################################################################
############################## BREAK POINT FOR MACC #################################################
#####################################################################################################

    # print('{},{:.2f},{:.2f},{:.2f},{}{:.2f}{}'.format(room_name, 
    #                                                      nostride_acc*100,
    #                                                      stride_acc*100, 
    #                                                      corrected_acc*100,
    #                                                      bcolors.OKGREEN,improvement*100,bcolors.ENDC))

#####################################################################################################
############################## BREAK POINT FOR MIOU #################################################
#####################################################################################################
    
    
    nostride_pred = ply_get_field(nostride_ply,'pred')
    nostride_gt = ply_get_field(nostride_ply,'gt')

    stride_pred = ply_get_field(stride_ply,'pred')
    stride_gt = ply_get_field(stride_ply,'gt')

    corrected_pred = ply_get_field(corrected_ply,'pred')
    corrected_gt = ply_get_field(corrected_ply,'gt')

    nostride_iou = metrics.jaccard_score(nostride_gt, nostride_pred, average = 'macro')
    stride_iou = metrics.jaccard_score(stride_gt, stride_pred, average = 'macro')
    corrected_iou = metrics.jaccard_score(corrected_gt, corrected_pred, average = 'macro')


    improvement_iou = corrected_iou - np.mean([nostride_iou,stride_iou])

    print('{},{:.2f},{:.2f},{:.2f},{}{:.2f}{}'.format(room_name, 
                                                         nostride_iou*100,
                                                         stride_iou*100, 
                                                         corrected_iou*100,
                                                         bcolors.OKGREEN,improvement_iou*100,bcolors.ENDC))

#####################################################################################################
############################### BREAK POINT FOR CAL CRV##############################################
#####################################################################################################
#     pcd_ply = corrected_ply
    
    
#     pcd_upred = ply_get_field(pcd_ply,'u_pred')    
#     pcd_upred = 1 - rescale_array(pcd_upred,0.01,.99)

#     pcd = ply_to_npy(pcd_ply)
    
#     # plt.hist(pcd_upred)
#     # plt.show()
    
#     bin_size = 0.01
#     bin_indx = (np.floor(pcd_upred / bin_size )).astype(int)
#     # print('max bin: {}, min bin: {}'.format(bin_indx.max(), bin_indx.min()))
#     # print('n bins: {}'.format(np.ceil(1/bin_size).astype(int)))

    
#     x = []
#     obs_probs = []
    
#     for bin in range(np.ceil(1/bin_size).astype(int)):
#         # print('bin: {}'.format(bin*bin_size))
#         points_in_bin = pcd[np.where(bin==bin_indx)]
#         obs_prob = 1-np.mean(points_in_bin[:,11])
#         x.append(bin*bin_size)
#         obs_probs.append(obs_prob)
        
#     x = np.array(x)
#     if cal_crv_plotted == False:
#         plt.plot(x,x/(1-bin_size),color='black', linestyle='--',label='perfectly calibrated')
#         cal_crv = x/(1-bin_size)
#         cal_crv_plotted = True
#         x_s_graph = x
#         plt.plot(x,obs_probs, color='blue',alpha=0.1, label = 'sample accuracy') # plot probs with label
#     else:
#         plt.plot(x,obs_probs, color='blue',alpha=0.03) # plot probs without label
#     obs_probs_all.append(obs_probs)
    
# obs_probs_all = np.array(obs_probs_all)
# obs_probs_all = np.nan_to_num(obs_probs_all)
# print(np.sum(obs_probs_all, axis = 0))
# plt.plot(x_s_graph, np.mean(obs_probs_all, axis = 0),color='red',label = 'mean accuracy')
# plt.plot(x_s_graph,np.percentile(obs_probs_all,5,axis=0),color = 'red',linestyle='--',alpha=0.5,label='5th and 95th percentile')
# plt.plot(x_s_graph,np.percentile(obs_probs_all,95,axis=0),color = 'red',linestyle='--',alpha=0.5)
# plt.fill_between(x_s_graph,np.mean(obs_probs_all, axis = 0),cal_crv,alpha=0.5,label = 'calibration error')
# plt.legend(prop = FONT_LEGEND)
# plt.xlabel('Predicted probability',fontsize=12, **FONT)
# plt.ylabel('Accuracy', fontsize=12 ,**FONT)

# plt.show()


#####################################################################################################
############################# BREAK POINT FOR IOU PER CLASS #########################################
#####################################################################################################    


    pcd_ply = stride_ply
    pred = ply_get_field(pcd_ply, 'pred')
    gt = ply_get_field(pcd_ply, 'gt')

    iou_container = [[] for _ in range(NUM_CATEGORIES)]
    iou = np.zeros(shape = (NUM_CATEGORIES)) 
    for cat in range(NUM_CATEGORIES):
        tp = np.array(list(zip(pred == gt , gt == cat))).all(axis = -1).sum() # true positives
        fp = np.array(list(zip(pred != gt , pred == cat))).all(axis = -1).sum() # false positives
        fn = np.array(list(zip(pred != gt , gt == cat))).all(axis = -1).sum() # false negatives
        
        # print(tp, fp, fn)
        iou = tp/(tp+fp+fn)
        if not np.isnan(iou) or iou == 0:
            iou_container[cat].append(iou)
    
    

ious = [0 for _ in range(NUM_CATEGORIES)]
for cat in range(NUM_CATEGORIES):
    ious[cat] = np.array(iou_container[cat]).mean()*100


print(ious)

#####################################################################################################
############################# BREAK POINT NUM POINTS IN CLASS #######################################
#####################################################################################################    

#     pcd_ply = corrected_ply
    
#     gt = ply_get_field(pcd_ply, 'gt')

#     cat_n_container = [[] for _ in range(NUM_CATEGORIES)]
#     iou = np.zeros(shape = (NUM_CATEGORIES)) 
#     for cat in range(NUM_CATEGORIES):
#         cat_n = np.array(np.where((gt == cat),True,False)).sum()
#         # print(np.where((gt == cat),True,False))

#         cat_n_container[cat] = cat_n
    
    

# # cat_n_container = np.array(cat_n_container)
# # cat_n_container = cat_n_container/cat_n_container.sum()
# print(cat_n_container)



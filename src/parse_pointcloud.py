import numpy as np
import open3d as o3d
from open3d import utility
import time

from visualize import bcolors
 

# draw axis-aligned bounding box from pointcloud
def draw_pcd_bbox(pcd):
    if isinstance(pcd,np.ndarray): points = pcd
    else: points = pcd.points 
    min_x,min_y,min_z = np.min(points, axis = 0)
    max_x,max_y,max_z = np.max(points, axis = 0)
    corner_box = np.array([ [min_x,min_y,min_z],
                            [max_x,min_y,min_z],
                            [max_x,max_y,min_z],
                            [min_x,max_y,min_z],
                            [min_x,min_y,max_z],
                            [max_x,min_y,max_z],
                            [max_x,max_y,max_z],
                            [min_x,max_y,max_z]])
    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
            [4, 5], [5, 6], [6, 7], [4, 7],
            [0, 4], [1, 5], [2, 6], [3, 7]]

    # Use the same color for all lines
    colors = [[.1, .1, 0] for _ in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corner_box)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

# get RGB vector from one-hot probabilities
def get_color_from_pred(pred, color_map):
    #n_classes = pred.shape[-1]
    pred = np.argmax(pred,axis = -1)
    colors = np.zeros(pred.shape + (3,), dtype = float)
    # print('call to get_color_from_pred with (pred) of shape: ',pred.shape)
    colors = np.vectorize(color_map.get,signature='()->(n)')(pred)
    return colors


#voxel-base sub-sampling
def one_point_per_voxel(points, voxel_size = 0.05):

    points_at_zero = points-np.min(points,axis=0)
    non_empty_voxel_keys,inverse,nb_points_per_voxel = np.unique(

                                                                (points_at_zero // voxel_size).astype(int), 
                                                                axis = 0,
                                                                return_inverse=True,
                                                                return_counts=True
                                                                )

    indx_pts_vox_sorted = np.argsort(inverse)
    voxel_grid = {} # dictionary to hold points
    grid_barycenter = []
    last_seen = 0
    for idx, vox in enumerate(non_empty_voxel_keys):
        voxel_grid[tuple(vox)] = points[indx_pts_vox_sorted
                                        [last_seen:last_seen+nb_points_per_voxel[idx]]]
        grid_barycenter.append(np.mean((voxel_grid[tuple(vox)]),axis=0))
        last_seen += nb_points_per_voxel[idx]
    return np.copy(grid_barycenter)    

def pcd_corner_to_zero(points): 
    '''returns translated pointcloud, where the leftmost bottom corner is as 0,0,0'''
    xyz = points[:,:3]
    features = points [:,3:]
    xyz_at_zero = xyz - np.min(xyz, axis=0)
    points_at_zero = np.concatenate([xyz_at_zero, features], axis = 1)
    return points_at_zero


def get_2d_voxels_old(points, voxel_size = 1, n_points_in_voxel = 4096): # n_points in original pointnet is 4096
    '''
    --old version--
    divide point cloud into vertical chunks to feed into the NN model
    input shape = N x FEATURES
    output shape = BINS x 4096 x FEATURES
    '''

    n_features = points.shape[-1]
    points_xyz = points[:,:3]
    points_at_zero = points_xyz-np.min(points_xyz,axis=0) # move corner of all points to origin
    voxels = (np.ceil(points_at_zero[:,:2]/voxel_size)).astype(int) # find the voxel bin id of each point
    x_bins, y_bins = np.max(voxels, axis = 0) # find number of voxels in x- and y
    binned_points = np.zeros((x_bins,y_bins,n_points_in_voxel,n_features),dtype = np.float32) # container for the points
    
    voxel_geometry = [] # this holds a mesh to visualize voxele boarders
    print('Found voxels: ', binned_points.shape[:2])
    for x_bin in range(x_bins):
         for y_bin in range(y_bins):
            ind_in_voxel = np.where(np.all(voxels==[x_bin+1,y_bin+1], axis = 1)) 
            points_in_voxel = points[ind_in_voxel]
            np.random.shuffle(points_in_voxel)
            


            if points_in_voxel.shape[0] != 0 : # check if there are any points in the voxel
                
                print('x: {}, y: {}, points: {}'.format(x_bin,y_bin, points_in_voxel.shape), end='\r')                                        
                points_in_voxel = np.take(points_in_voxel,range(n_points_in_voxel),mode='wrap',axis = 0)
                binned_points[x_bin, y_bin,:,:] = points_in_voxel
                
            #draw voxel boundary (for debugging)
            voxel_geometry.append(o3d.geometry.\
                TriangleMesh.create_box(voxel_size,voxel_size,0.001).translate(
                    np.min(points_xyz,axis=0) + np.array([x_bin*voxel_size,y_bin*voxel_size,-0.1])
                ))

    #flatten bin dimension and remove empty voxels (if any)
    binned_points = binned_points.reshape((-1,binned_points.shape[-2], binned_points.shape[-1]))
    bin_mean = np.all(np.mean(binned_points,axis=1).astype(bool), axis = 1)
    binned_points = binned_points[bin_mean,...]             
                
    return binned_points, voxel_geometry


def get_2d_voxels(points, voxel_size = 1, n_points_in_voxel = 4096, normalized = True, show_boundary = False): # n_points in original pointnet is 4096
    '''
    divide point cloud into vertical chunks to feed into the NN model
    input shape = N x FEATURES
    output shape = BINS x 4096 x FEATURES
    
    input shape = N x 7               ---> features: [x, y, z, r, g, b, label]
    output shape = BINS x 4096 x 10   ---> features: [x, y, z, r, g, b, x_loc, y_loc, z_loc, label
    '''
    # input shape = N x 7               ---> features: [x, y, z, r, g, b, label]
    # output shape = BINS x 4096 x 10   ---> features: [x, y, z, r, g, b, x_loc, y_loc, z_loc, label]
    
    n_features = points.shape[-1]
    points_xyz = points[:,:3]
    points_at_zero = points_xyz#-np.min(points_xyz,axis=0) # move corner of all points to origin
    if normalized : # add 3 features for the normalized location relative to scene
        n_features += 3
        points_norm_location = points_at_zero / np.max (points_at_zero, axis = 0) # normalized relative position of points to scene
        points = np.append(points,points_norm_location, axis=1)
        points = points[:,[*list(range(6)),7,8,9,6]] # shift label to the end of the vector
    voxels = (np.ceil(points_at_zero[:,:2]/voxel_size)).astype(int) # find the voxel bin id of each point
    x_bins, y_bins = np.max(voxels, axis = 0) # find number of voxels in x- and y
    binned_points = np.zeros((x_bins,y_bins,n_points_in_voxel,n_features),dtype = np.float32) # container for the points
    
    start = time.time()
    voxel_geometry = [] # this holds a mesh to visualize voxele boarders
    voxel_origins = [] # container for all voxel origin coordinates for later reconstruction
    n_all_vox = np.prod(binned_points.shape[:2])
    n_empty_vox = 0
    print('Found {} voxels with bin shape: {}'.format(n_all_vox, binned_points.shape[:2]))
    for x_bin in range(x_bins):
         for y_bin in range(y_bins):
            ind_in_voxel = np.where(np.all(voxels==[x_bin+1,y_bin+1], axis = 1)) 
            points_in_voxel = points[ind_in_voxel]
            np.random.shuffle(points_in_voxel)
            


            if points_in_voxel.shape[0] != 0 : # check if there are any points in the voxel
                
                print(" "*40,end = '\r')
                print('x: {}, y: {}, points: {}'.format(x_bin,y_bin, points_in_voxel.shape), end='\r')
                
                if normalized:
                    
                    
                    # find origin and move points to it
                    voxel_origin = np.min(points_in_voxel[:,:3], axis=0) + [0.5,0.5,0]
                    voxel_origins.append(voxel_origin)
                    #reconstruct features vector
                    points_in_voxel = np.append(points_in_voxel[:,:3]-voxel_origin, 
                                                points_in_voxel[:,3:]
                                                ,axis=-1)
                    
                    
                points_in_voxel = np.take(points_in_voxel,range(n_points_in_voxel),mode='wrap',axis = 0) # sample n random points in block/voxel
                binned_points[x_bin, y_bin,:,:] = points_in_voxel
            else: 
                # print ('{}WARNING: Empty voxel detected!{}'.format(bcolors.WARNING,bcolors.ENDC))
                # voxel_origins.append(np.array((0,0,0),dtype = np.float32))
                voxel_origins.append("Empty")
                n_empty_vox += 1
            #draw voxel boundary (for debugging)
            if show_boundary:
                voxel_geometry.append(o3d.geometry.\
                    TriangleMesh.create_box(voxel_size,voxel_size,0.001).translate(
                        np.min(points_xyz,axis=0) + np.array([x_bin*voxel_size,y_bin*voxel_size,-0.1])
                    ))

    #flatten bin dimension 
    binned_points = binned_points.reshape((-1,binned_points.shape[-2], binned_points.shape[-1])) # block , n_points in block,  n_features
    
    #remove empty voxels
    binned_points_ne = np.zeros([0,*list(binned_points.shape[1:])])
    voxel_origins_ne = []
    voxel_geometry_ne = []
    for i in range(n_all_vox):
        if voxel_origins[i] != "Empty":
            binned_points_ne = np.append(binned_points_ne, binned_points[i,...][np.newaxis,...],axis=0)
            voxel_origins_ne.append(voxel_origins[i])
            if show_boundary : voxel_geometry_ne.append(voxel_geometry[i])



    print('\nRetrieved {} voxels in: {:0.2f} seconds\n{}{} Voxels were empty{}'.format(binned_points_ne.shape[0],time.time()-start,bcolors.OKGREEN,n_empty_vox,bcolors.ENDC))                
    return binned_points_ne, voxel_origins_ne, voxel_geometry_ne


def get_3d_voxels(points, voxel_size = [[2.5,2.5,5]], n_points_in_voxel = 4096, normalized = True, show_boundary = False): # n_points in original pointnet is 4096
    '''
    divide point cloud into blocks of a given size to feed into the NN model
    the function assumes the pointcloud has no negative xyz!
    input shape = N x FEATURES
    output shape = BINS x 4096 x FEATURES
    '''
    # input shape = N x 7               ---> features: [x, y, z, r, g, b, label]
    # output shape = BINS x 4096 x 10   ---> features: [x, y, z, r, g, b, x_loc, y_loc, z_loc, label]
    
    n_features = points.shape[-1]
    points_xyz = points[:,:3]
    points_at_zero = points_xyz#-np.min(points_xyz,axis=0) # move corner of all points to origin
    if normalized : # add 3 features for the normalized location relative to scene
        n_features += 3
        points_norm_location = points_at_zero / np.max (points_at_zero, axis = 0) # normalized relative position of points to scene
        points = np.append(points,points_norm_location, axis=1)
        points = points[:,[*list(range(6)),7,8,9,6]] # shift label to the end of the vector
    voxels = (np.ceil(points_at_zero[:,:3]/voxel_size)).astype(int) # find the voxel bin id of each point
    x_bins, y_bins, z_bins = np.max(voxels, axis = 0) # find number of voxels in x,y and z
    binned_points = np.zeros((x_bins,y_bins,z_bins,n_points_in_voxel,n_features),dtype = np.float32) # container for the points
    
    start = time.time()
    voxel_geometry = [] # this holds a mesh to visualize voxele boarders
    voxel_origins = [] # container for all voxel origin coordinates for later reconstruction
    n_all_vox = np.prod(binned_points.shape[:3])
    n_empty_vox = 0
    print('Found {} voxels with bin shape: {}'.format(n_all_vox, binned_points.shape[:3]))
    for x_bin in range(x_bins):
         for y_bin in range(y_bins):
             for z_bin in range(z_bins):
                ind_in_voxel = np.where(np.all(voxels==[x_bin+1,y_bin+1,z_bin+1], axis = 1)) 
                points_in_voxel = points[ind_in_voxel]
                np.random.shuffle(points_in_voxel)
                


                if points_in_voxel.shape[0] > 250 : # check if there are any points in the voxel
                    
                    print(" "*40,end = '\r')
                    print('x: {}, y: {},z:{}, points: {}'.format(x_bin,y_bin,z_bin, points_in_voxel.shape), end='\r')
                    
                    if normalized:
                        
                        
                        # find origin and move points to it
                        voxel_origin = np.min(points_in_voxel[:,:3], axis=0) + [0.5,0.5,0]
                        voxel_origins.append(voxel_origin)
                        #reconstruct features vector
                        points_in_voxel = np.append(points_in_voxel[:,:3]-voxel_origin, 
                                                    points_in_voxel[:,3:]
                                                    ,axis=-1)
                        
                        
                    points_in_voxel = np.take(points_in_voxel,range(n_points_in_voxel),mode='wrap',axis = 0) # sample n random points in block/voxel
                    binned_points[x_bin, y_bin,z_bin,:,:] = points_in_voxel
                else: 
                    # print ('{}WARNING: Empty voxel detected!{}'.format(bcolors.WARNING,bcolors.ENDC))
                    # voxel_origins.append(np.array((0,0,0),dtype = np.float32))
                    voxel_origins.append("Empty")
                    n_empty_vox += 1
                #draw voxel boundary (for debugging)
                if show_boundary:
                    voxel_geometry.append(o3d.geometry.\
                        TriangleMesh.create_box(voxel_size,voxel_size,0.001).translate(
                            np.min(points_xyz,axis=0) + np.array([x_bin*voxel_size[0],y_bin*voxel_size[1],-0.1])
                        ))

    #flatten bin dimension 
    binned_points = binned_points.reshape((-1,binned_points.shape[-2], binned_points.shape[-1])) # block , n_points in block,  n_features
    
    #remove empty voxels
    binned_points_ne = np.zeros([0,*list(binned_points.shape[1:])])
    voxel_origins_ne = []
    voxel_geometry_ne = []
    for i in range(n_all_vox):
        if voxel_origins[i] != "Empty":
            binned_points_ne = np.append(binned_points_ne, binned_points[i,...][np.newaxis,...],axis=0)
            voxel_origins_ne.append(voxel_origins[i])
            if show_boundary : voxel_geometry_ne.append(voxel_geometry[i])



    print('\nRetrieved {} voxels in: {:0.2f} seconds\n{}{} Voxels were empty{}'.format(binned_points_ne.shape[0],time.time()-start,bcolors.OKGREEN,n_empty_vox,bcolors.ENDC))                
    return binned_points_ne, voxel_origins_ne, voxel_geometry_ne
if __name__ == '__main__':
    points_colors_labels = np.load('./data/temp_npy/Area_2/train/hallway_3.npy')
    points,boxes = get_2d_voxels(points_colors_labels,normalized = True)
    print('output shape is :', points.shape)   

    n = 15 # display one voxel

    colors = points[:,:,3:6].reshape(-1,3)/255.
    feature_colors = np.repeat(points[:,:,-1],axis = -1, repeats = 3).reshape(-1,3)/(3.,9.,24.)
    points = points[:,:,6:9].reshape(-1,3)
    print('max output: ', np.max(points,axis = 0))
    #render

    filtered_pointcloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[:,:3]))
    filtered_pointcloud.colors = o3d.utility.Vector3dVector(colors)

    bbox = draw_pcd_bbox(filtered_pointcloud)


    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0,origin = [0.,0.,0.])
    render_items = [bbox,filtered_pointcloud,*boxes]
    render_origin = True
    if render_origin:
        render_items.append(origin)
    o3d.visualization.draw_geometries(render_items, mesh_show_wireframe = True)
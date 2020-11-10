import os 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shdom
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable, AxesGrid

import scipy.io as sio

from mayavi import mlab


def FindOutPoint(start_point,direction,grid):
    """
    Find the intersection of the ray with the grid.
    BUT here the origin is inside the grid.
    
    Input:
       start_point -  - np.array. ray origin
       direction - np.array. ray direction
       grid: shdom.grid.
    """
    
    xmin = grid.bounding_box.xmin
    xmax = grid.bounding_box.xmax
    
    ymin = grid.bounding_box.ymin
    ymax = grid.bounding_box.ymax
    
    zmin = grid.bounding_box.zmin
    zmax = grid.bounding_box.zmax  
    
    dx = grid.dx
    dy = grid.dy
    Lz = zmax - zmin
    dz = Lz/(grid.nz-1)      
    # The _max is one d_ befor the last corner (|_|_|_|_|_|_|_->|_|).
    # important step, I always miss this point.
    xmax += dx
    ymax += dy
    zmax += dz
    
    # The _max is one d_ befor the last corner (|_|_|_|_|_|_|_->|_|).
    minBound = np.array([xmin, ymin, zmin])
    maxBound = np.array([xmax, ymax, zmax])
    boxSize = maxBound - minBound
    
    CX = (xmin <= start_point[0]) and (xmax >= start_point[0])
    CY = (ymin <= start_point[1]) and (ymax >= start_point[1])
    CZ = (zmin <= start_point[2]) and (zmax >= start_point[2])
    assert (CX and CY and CZ) , "This function works only if the point is inside the grid."
    # starting point is inside the grid:
     
    if (direction[0]>=0):
        tVoxelX = 1
    else:
        tVoxelX = 0
    
    if (direction[1]>=0):
        tVoxelY = 1
    else:
        tVoxelY = 0
    
    if (direction[2]>=0):
        tVoxelZ = 1
    else:
        tVoxelZ = 0
    
    boxMaxX  = minBound[0] + tVoxelX*boxSize[0]
    boxMaxY  = minBound[1] + tVoxelY*boxSize[1]
    boxMaxZ  = minBound[2] + tVoxelZ*boxSize[2]
    
    tMaxX    =  (boxMaxX-start_point[0])/direction[0]
    tMaxY    =  (boxMaxY-start_point[1])/direction[1]
    tMaxZ    =  (boxMaxZ-start_point[2])/direction[2]
    
    if(abs(np.arccos(direction[2]) - np.pi) >= 1e-6):
        
        if (tMaxX < tMaxY):
            if (tMaxX < tMaxZ):
                s = tMaxX 
            else:
                s = tMaxZ
        else:
            if (tMaxY < tMaxZ):
                s = tMaxY 
            else:
                s = tMaxZ 
    
    else:
        s = tMaxZ
        
    return start_point + s*direction;

def LengthRayInGrid(start_point,direction,grid, Intersection = False):
    """
    A fast and simple voxel traversal algorithm through a 3D space partition (grid)
    proposed by J. Amanatides and A. Woo (1987).
    
    Based on: https://www.mathworks.com/matlabcentral/fileexchange/26852-a-fast-voxel-traversal-algorithm-for-ray-tracing
    
    Input:
       start_point -  - np.array. ray origin
       direction - np.array. ray direction
       grid: shdom.grid.
       Intersection - bool, if True returns also the Intersection point eith the grid bounding box.
    """
    SHOW_TRAVEL = False # for debugging porpuse
    nx = grid.nx
    ny = grid.ny
    nz = grid.nz
    
    xmin = grid.bounding_box.xmin
    xmax = grid.bounding_box.xmax
    
    ymin = grid.bounding_box.ymin
    ymax = grid.bounding_box.ymax
    
    zmin = grid.bounding_box.zmin
    zmax = grid.bounding_box.zmax  
    
    dx = grid.dx
    dy = grid.dy
    Lz = zmax - zmin
    dz = Lz/(nz-1)      
    # The _max is one d_ befor the last corner (|_|_|_|_|_|_|_->|_|).
    
    xgrid = np.linspace(xmin, xmax,nx)
    ygrid = np.linspace(ymin, ymax,ny)
    zgrid = np.linspace(zmin, zmax,nz)   
    
    X, Y, Z = np.meshgrid(xgrid, ygrid, zgrid, indexing='ij')
    # important step, I always miss this point.
    xmax += dx
    ymax += dy
    zmax += dz
    
    minBound = np.array([xmin, ymin, zmin])
    maxBound = np.array([xmax, ymax, zmax])
    boxSize = maxBound - minBound
    
    voxelSizeX = boxSize[0]/nx
    voxelSizeY = boxSize[1]/ny
    voxelSizeZ = boxSize[2]/nz   
    
    if(SHOW_TRAVEL):
        
        scale_factor = 1
        if(0):
             
            # show bounding_box
            xm = [xmin, xmax, xmax, xmin, xmax, xmax, xmin, xmin ]
            ym = [ymin, ymin, ymin, ymin, ymax, ymax, ymax, ymax ]
            zm = [zmin, zmin, zmax, zmax, zmin, zmax, zmax, zmin ]
            # Medium cube
            triangles = [[0,1,2],[0,3,2],[1,2,5],[1,4,5],[2,5,6],[2,3,6],[4,7,6],[4,5,6],[0,3,6],[0,7,6],[0,1,4],[0,7,4]];
            figh = mlab.gcf()
            obj = mlab.triangular_mesh( xm, ym, zm, triangles,color = (0.0, 0.17, 0.72),opacity=0.1,figure=figh)
            figh.scene.anti_aliasing_frames = 0
            
    # find ray medium intersection:
    CX = (xmin <= start_point[0]) and (xmax >= start_point[0])
    CY = (ymin <= start_point[1]) and (ymax >= start_point[1])
    CZ = (zmin <= start_point[2]) and (zmax >= start_point[2])
    
    origin_x, origin_y, origin_z = start_point
    d_x, d_y, d_z = direction
    
    if(CX and CY and CZ):
        # starting point is inside the grid:
        print('start point is in the grid')
    
    else:
        # first find the intersection of the ray with the grid:
        # Ray/box intersection using the Smits' algorithm.
        # Based on https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
        
        if(d_x >= 0):
            tmin = (xmin - origin_x) / d_x
            tmax = (xmax - origin_x) / d_x
        else:
            tmin = (xmax - origin_x) / d_x
            tmax = (xmin - origin_x) / d_x

        if(tmin>tmax):
            tmin, tmax = tmax, tmin # swap

        if(d_y >= 0):
            tmin2 = (ymin - origin_y) / d_y
            tmax2 = (ymax - origin_y) / d_y
        else:
            tmin2 = (ymax - origin_y) / d_y
            tmax2 = (ymin - origin_y) / d_y

        if(tmin2>tmax2):
            tmin2, tmax2 = tmax2, tmin2 # swap

        if ( (tmin > tmax2) or (tmin2 > tmax) ):
            # if we here, there is no intersection.
            if(Intersection):
                return -1, -1, None
            else:
                return -1, -1

        if (tmin2 > tmin):
            tmin = tmin2

        if (tmax2 < tmax): 
            tmax = tmax2


        if(d_z >= 0):
            tmin2 = (zmin - origin_z) / d_z
            tmax2 = (zmax - origin_z) / d_z
        else:
            tmin2 = (zmax - origin_z) / d_z
            tmax2 = (zmin - origin_z) / d_z

        if(tmin2>tmax2):
            tmin2, tmax2 = tmax2, tmin2 # swap

        if ( (tmin > tmax2) or (tmin2 > tmax) ):
            # if we here, there is no intersection.
            if(Intersection):
                return -1, -1, None
            else:
                return -1, -1

        if (tmin2 > tmin):
            tmin = tmin2

        if (tmax2 < tmax): 
            tmax = tmax2


        inte_bbox_x = origin_x + tmin*d_x
        inte_bbox_y = origin_y + tmin*d_y
        inte_bbox_z = origin_z + tmin*d_z
        inte_length = tmin
        bbox_intersection_point = np.array([inte_bbox_x,inte_bbox_y,inte_bbox_z])
        
        if(SHOW_TRAVEL):
            # show travel from origin to the nearest ntersection:
            mlab.points3d(inte_bbox_x, inte_bbox_y, inte_bbox_z, 
                          scale_factor=0.006,color=(1,0,0))
            
            
            # scale_factor=2*scale
            
            #mlab.quiver3d(origin_x, origin_z, origin_z, inte_length*d_x, 
                                      #inte_length*d_y, inte_length*d_z,\
                                      #line_width=0.001,color = (0.6,1.0,0),opacity=0.2,mode='2ddash',scale_factor=scale_factor)#,scale_factor=2*scale 
        
        # Avoide numerical issue of inaccurate intersection:
        CX = (xmin <= bbox_intersection_point[0]) and (xmax >= bbox_intersection_point[0])
        CY = (ymin <= bbox_intersection_point[1]) and (ymax >= bbox_intersection_point[1])
        CZ = (zmin <= bbox_intersection_point[2]) and (zmax >= bbox_intersection_point[2])
        C = CX and CY and CZ
        eps = 1e-6
        error_counter = 0
        while(not C):
            bbox_intersection_point += eps*np.array([d_x,d_y,d_z])
            error_counter += eps
            
            CX = (xmin <= bbox_intersection_point[0]) and (xmax >= bbox_intersection_point[0])
            CY = (ymin <= bbox_intersection_point[1]) and (ymax >= bbox_intersection_point[1])
            CZ = (zmin <= bbox_intersection_point[2]) and (zmax >= bbox_intersection_point[2])
            C = CX and CY and CZ
            
        assert error_counter < 1e-3, "The intersection point seems to be too far from the bounding box, {}".format(error_counter)    
        start_point = bbox_intersection_point + 1e-6*np.array([d_x,d_y,d_z])
        
    end_point = FindOutPoint(start_point,direction,grid)
    totalLength = np.linalg.norm(end_point - start_point) # total travel path in the grid.
    
    if(SHOW_TRAVEL):
        # show out point:
        mlab.points3d(end_point[0], end_point[1], end_point[2], 
                      scale_factor=0.006,color=(1,0,0))
        
        mlab.quiver3d(start_point[0], start_point[1], start_point[2],
                      totalLength*d_x, 
                      totalLength*d_y, totalLength*d_z,\
                      line_width=0.001,color = (0.1,0.2,0),opacity=0.2,mode='2ddash',scale_factor=scale_factor)        
        
    # -------------------------------------------------------------
    # --------------- find intersections: --------------------------
    # -------------------------------------------------------------
    
    # calculate the start point voxel index
    index_x = np.floor( ((start_point[0] - minBound[0])/boxSize[0])*nx )
    index_y = np.floor( ((start_point[1] - minBound[1])/boxSize[1])*ny )
    index_z = np.floor( ((start_point[2] - minBound[2])/boxSize[2])*nz )
    # check if we inside the grid as we should:
    CX = (0 <= index_x) and (nx > index_x)
    CY = (0 <= index_y) and (ny > index_y)
    CZ = (0 <= index_z) and (nz > index_z)   
    
    assert CX and CY and CZ ,"voxel is out of the grid {}, {}, {}".format(index_x, index_y, index_z)
    
    # path related to voxel boundry in j [from (x,y,z)] axis
    tDeltaX    = voxelSizeX/abs(d_x)
    tDeltaY    = voxelSizeY/abs(d_y)
    tDeltaZ    = voxelSizeZ/abs(d_z)
    # calculate step orintation
    if (d_x >= 0):
        tVoxelX = (index_x+1)/nx
        stepX = 1
    else:
        tVoxelX = (index_x)/nx
        stepX = -1
    
    
    if (d_y >= 0):
        tVoxelY = (index_y+1)/ny
        stepY = 1
    else:
        tVoxelY = (index_y)/ny
        stepY = -1
    
    if (d_z >= 0):
        tVoxelZ = (index_z+1)/nz
        stepZ = 1
    else:
        tVoxelZ = (index_z)/nz
        stepZ = -1
    
    # calculate location of the cell fase in original dimensions (x y z) = (voxelMaxX , voxelMaxY , voxelMaxZ ).
    voxelMaxX  = minBound[0] + tVoxelX*boxSize[0]
    voxelMaxY  = minBound[1] + tVoxelY*boxSize[1]
    voxelMaxZ  = minBound[2] + tVoxelZ*boxSize[2]
    # Only initiat path in j [from (x,y,z)] axis . and next these variables
    # will be path counter in [x y z] directions.
    tMaxX      =  (voxelMaxX - start_point[0])/d_x
    tMaxY      =  (voxelMaxY - start_point[1])/d_y
    tMaxZ      =  (voxelMaxZ - start_point[2])/d_z

    vmin_visualizatio = start_point
    # -----------------------------------------------
    # starting the loop:
    ref = 0 
    path_length = 0
    lengths = [ ] 
    Vindx = [ ]
    Vindy = [ ]
    Vindz = [ ]
    # Vindx - is voxel index in x axis.
    x , y, z = start_point 
    
    TRAVEL = True
    
    while(TRAVEL):
        old_index_x = int(index_x)
        old_index_y = int(index_y)
        old_index_z = int(index_z)
        
        Vindx.append(int(index_x))
        Vindy.append(int(index_y))
        Vindz.append(int(index_z))
        
        if(abs(np.arccos(direction[2]) - np.pi) >= 1e-6):
            if (tMaxX < tMaxY):
                
                if (tMaxX < tMaxZ):
                    
                    s = tMaxX - ref  
                    index_x = index_x + stepX
                    path_length = path_length + s
                    ref =  tMaxX
                    tMaxX = tMaxX + tDeltaX
                else:
                    
                    s = tMaxZ - ref 
                    index_z = index_z + stepZ
                    path_length = path_length + s
                    ref = tMaxZ 
                    tMaxZ = tMaxZ + tDeltaZ
                    
            else:
                
                if (tMaxY < tMaxZ):
                    
                    s = tMaxY - ref
                    index_y = index_y + stepY
                    path_length = path_length + s
                    ref =  tMaxY 
                    tMaxY = tMaxY + tDeltaY
                    
                else:
                    
                    s = tMaxZ - ref
                    index_z = index_z + stepZ
                    path_length = path_length + s
                    ref =  tMaxZ 
                    tMaxZ = tMaxZ + tDeltaZ
                    
        else:# e.g direction is 0,0,-1
            s = tMaxZ - ref
            index_z = index_z + stepZ
            path_length = path_length + s
            ref =  tMaxZ 
            tMaxZ = tMaxZ + tDeltaZ

                        
        if (( totalLength - path_length) <= 1e-6):
            errs = path_length - totalLength
            s = s - errs
            TRAVEL = False # stop the loop
            
        lengths.append(s)
        
        
        x = x + d_x * s 
        y = y + d_y * s 
        z = z + d_z * s 
                
        if(SHOW_TRAVEL):
            
            # show current point on grid:
            mlab.points3d(x, y, z, 
                          scale_factor=0.003,color=(0,0,0))
            
            # show voxel box
            vmin_visualizatio = np.array([(old_index_x)/nx, (old_index_y)/ny, (old_index_z)/nz ])
            vmin_visualizatio = minBound + vmin_visualizatio*boxSize
            xmin_vis, ymin_vis, zmin_vis = vmin_visualizatio
            vmax_visualizatio = np.array([(old_index_x+1)/nx, (old_index_y+1)/ny, (old_index_z+1)/nz ])
            vmax_visualizatio = minBound + vmax_visualizatio*boxSize
            xmax_vis, ymax_vis, zmax_vis = vmax_visualizatio
            
            xm = [xmin_vis, xmax_vis, xmax_vis, xmin_vis, xmax_vis, xmax_vis, xmin_vis, xmin_vis ]
            ym = [ymin_vis, ymin_vis, ymin_vis, ymin_vis, ymax_vis, ymax_vis, ymax_vis, ymax_vis ]
            zm = [zmin_vis, zmin_vis, zmax_vis, zmax_vis, zmin_vis, zmax_vis, zmax_vis, zmin_vis ]
            # voxel cube
            if(0):
                
                triangles = [[0,1,2],[0,3,2],[1,2,5],[1,4,5],[2,5,6],[2,3,6],[4,7,6],[4,5,6],[0,3,6],[0,7,6],[0,1,4],[0,7,4]];
                obj = mlab.triangular_mesh( xm, ym, zm, triangles,color = (0.76, 0.17, 0.72),opacity=0.1,figure=figh)
                figh.scene.anti_aliasing_frames = 0
                
            
            
            
            #----------------------------------------------------------
    #mlab.show()
    voxels = np.ravel_multi_index([Vindx, Vindy, Vindz], [nx, ny, nz])
    # An array of indices into the flattened version of an array of [nx, ny, nz]
    if(Intersection):
        return voxels, lengths, start_point
    else:
        return voxels, lengths
  
  
  
  
  
def show_voxels(volume, grid):
    
    # https://stackoverflow.com/questions/37849063/how-are-x-y-z-coordinates-defined-for-mayavis-mlab-mesh
    voxel_indexes = np.argwhere(volume)
    voxel_indexes = voxel_indexes.tolist()
    
    nx = grid.nx
    ny = grid.ny
    nz = grid.nz
    
    xmin = grid.bounding_box.xmin
    xmax = grid.bounding_box.xmax
    
    ymin = grid.bounding_box.ymin
    ymax = grid.bounding_box.ymax
    
    zmin = grid.bounding_box.zmin
    zmax = grid.bounding_box.zmax  
    
    dx = grid.dx
    dy = grid.dy
    Lz = zmax - zmin
    dz = Lz/(nz-1)
    
    minBound = np.array([xmin, ymin, zmin])
    maxBound = np.array([xmax, ymax, zmax])
    boxSize = maxBound - minBound
    
    xgrid = np.linspace(xmin, xmax,nx)
    ygrid = np.linspace(ymin, ymax,ny)
    zgrid = np.linspace(zmin, zmax,nz)         
    X, Y, Z = np.meshgrid(xgrid, ygrid, zgrid, indexing='ij')
    
        
    #             pt1_ _ _ _ _ _ _ _ _pt2
    #              /|                 /|
    #             / |                / |
    #         pt3/_ | _ _ _ _ _ _pt4/  |
    #           |   |              |   |
    #           |   |              |   |
    #           |  pt5_ _ _ _ _ _ _|_ _|pt6
    #           |  /               |  /
    #           | /                | /
    #        pt7|/_ _ _ _ _ _ _ _ _|/pt8
    
    opacity = 0.5
    
    for index in voxel_indexes:
        
        # Where :
        side_point = np.array([X[index[0],index[1],index[2]], 
                                Y[index[0],index[1],index[2]], 
                                Z[index[0],index[1],index[2]]])
        
        x1, y1, z1 = side_point + np.array([0, dy, dz])  # | => pt1
        x2, y2, z2 = side_point + np.array([dx, dy, dz])  # | => pt2
        x3, y3, z3 = side_point + np.array([0, 0, dz])  # | => pt3
        x4, y4, z4 = side_point + np.array([dx, 0, dz])  # | => pt4
        x5, y5, z5 = side_point + np.array([0, dy, 0])  # | => pt5
        x6, y6, z6 = side_point + np.array([dx, dy, 0])  # | => pt6
        x7, y7, z7 = side_point + np.array([0, 0, 0])  # | => pt7
        x8, y8, z8 = side_point + np.array([dx, 0, 0])  # | => pt8        
        
        
        
        box_points = np.array([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3],
                                  [x4, y4, z4], [x5, y5, z5], [x6, y6, z6],
                                  [x7, y7, z7], [x8, y8, z8]],dtype=np.dtype(float))
        
        
        
        mlab.points3d(box_points[:, 0], box_points[:, 1], box_points[:, 2],
                             color=(1, 0, 0),scale_factor=0.003)
        
        
        mlab.mesh([[x1, x2],
                          [x3, x4]],  # | => x coordinate
        
                         [[y1, y2],
                          [y3, y4]],  # | => y coordinate
        
                         [[z1, z2],
                          [z3, z4]],  # | => z coordinate
        
                         color=(0, 0.1, 0.02), opacity=opacity)  # black
        
        # Where each point will be connected with this neighbors :
        # (link = -)
        #
        # x1 - x2     y1 - y2     z1 - z2 | =>  pt1 - pt2
        # -    -  and  -   -  and -    -  | =>   -     -
        # x3 - x4     y3 - y4     z3 - z4 | =>  pt3 - pt4
        
        
        mlab.mesh([[x5, x6], [x7, x8]],
                         [[y5, y6], [y7, y8]],
                         [[z5, z6], [z7, z8]],
                         color = (0., 0.1, 0.02), opacity=opacity)  # 
        
        mlab.mesh([[x1, x3], [x5, x7]],
                         [[y1, y3], [y5, y7]],
                         [[z1, z3], [z5, z7]],
                         color = (0., 0.1, 0.02), opacity=opacity)  # 
        
        mlab.mesh([[x1, x2], [x5, x6]],
                         [[y1, y2], [y5, y6]],
                         [[z1, z2], [z5, z6]],
                         color = (0., 0.1, 0.02), opacity=opacity)  # 
        
        mlab.mesh([[x2, x4], [x6, x8]],
                         [[y2, y4], [y6, y8]],
                         [[z2, z4], [z6, z8]],
                         color = (0., 0.1, 0.02), opacity=opacity)  # 
        
        mlab.mesh([[x3, x4], [x7, x8]],
                         [[y3, y4], [y7, y8]],
                         [[z3, z4], [z7, z8]],
                         color = (0., 0.1, 0.02), opacity=opacity)  # 
        
    #mlab.show()      
        
          
      
      
      
     

      
        


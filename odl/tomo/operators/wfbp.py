
from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import numpy as np

from odl.discr import DiscretizedSpace
from odl.tomo.geometry import Geometry

def wfbp(recon_space, geometry, proj_data):
    """
    First implementation of WFBP - At most 1 ray per angle!
    """
    angles = geometry.angles
    R = geometry.det_radius + geometry.src_radius

    V = np.zeros(recon_space.shape + angles.shape, dtype='float32', order='C')

    for ix,iy,iz in np.ndindex(recon_space.shape):
        x,y,z = recon_space.grid[ix,iy,iz]

        for itheta in range(angles.shape[0]):
            theta = angles[itheta]
            # get u, v - clockwise rotation by theta 
            u = x*np.cos(theta) + y*np.sin(theta)
            l = -x*np.sin(theta) + y*np.cos(theta) 
            z_src = theta * geometry.pitch / (2 * np.pi)
            v = (z - z_src) * R / (l + geometry.src_radius)

            iu = geometry.det_partition.byaxis[0].index(u)
            try:
                iv = geometry.det_partition.byaxis[1].index(v)
                V[ix,iy,iz, itheta] = proj_data[itheta,iu,iv]
            except:
                pass
            
    return np.sum(V, axis=-1)

def wfbp_theta(recon_space, geometry, proj_data):
    """
    Implementation of WFBP with vectorised computation on theta - TODO: multi book!
    """
    angles = geometry.angles
    zs_src = angles * geometry.pitch / (2 * np.pi)
    R = geometry.det_radius + geometry.src_radius

    u_min, v_min = geometry.det_partition.min_pt
    u_max, v_max = geometry.det_partition.max_pt
    u_cell, v_cell = geometry.det_partition.cell_sides
    

    V_tot = np.empty(recon_space.shape, dtype='float32', order='C')

    for ix,iy,iz in np.ndindex(recon_space.shape):
        x,y,z = recon_space.grid[ix,iy,iz]
        # get u,v for rays throug the voxel at each angle
        # clockwise rotation by angles
        us = x*np.cos(angles) + y*np.sin(angles)
        ls = -x*np.sin(angles) + y*np.cos(angles)
        vs = (z - zs_src) * R / (ls + geometry.src_radius)

        # add mask for rays hitting the detector
        mask  = vs > v_min
        mask *= vs < v_max

        us = us[mask]
        vs = vs[mask]

        ius = (us - u_min) // u_cell
        ius = ius.astype(np.int)
        ivs = (vs - v_min) // v_cell
        ivs = ivs.astype(np.int)

        ithetas = np.arange(angles.shape[0])[mask]

        V_tot[ix,iy,iz] = np.sum(proj_data[ithetas,ius,ivs])
    return V_tot

def wfbp_full(recon_space, geometry, proj_data):
    """
    Implementation of WFBP fully vectorised - TODO: requires a lot of memory
    This is just an exercise!
    """
    angles = geometry.angles
    zs_src = angles * geometry.pitch / (2 * np.pi)
    zs_src = zs_src.reshape((1,1,1,-1))
    R = geometry.det_radius + geometry.src_radius

    u_min, v_min = geometry.det_partition.min_pt
    u_max, v_max = geometry.det_partition.max_pt
    u_cell, v_cell = geometry.det_partition.cell_sides

    V_tot = np.empty(recon_space.shape, dtype='float32', order='C')

    x, y, z = recon_space.grid.coord_vectors

    # shapes: (x, 1, angles) + (1, y, angles) -> (x,y,angles)
    # given x,y,theta compute u = x*np.cos(theta) + y*np.sin(theta)
    us = (np.outer(x, np.cos(angles)).reshape((x.shape[0],1,1,angles.shape[0])) +
          np.outer(y, np.sin(angles)).reshape((1,y.shape[0],1,angles.shape[0])))
    ls = (np.outer(x, -np.sin(angles)).reshape((x.shape[0],1,1,angles.shape[0])) +
          np.outer(y, np.cos(angles)).reshape((1,y.shape[0],1,angles.shape[0])))

    # get grid indices
    ius = ((us - u_min) // u_cell).astype(np.int16)

    # split z to avoid memory problems:
    buffer_size = 16
    for i in range(0, z.shape[0], buffer_size):
        _z = z[i:i+buffer_size]
        vs = (_z.reshape((1,1,-1,1)) - zs_src) * R / (ls + geometry.src_radius)
    
        mask  = vs < v_min
        mask += vs > v_max

    
        ivs = ((vs - v_min) // v_cell).astype(np.int16)

        # Deal with ivs out of range
        _proj_data = np.pad(proj_data, [(0,0),(0,0),(0,1)], mode='constant')
        ivs[mask] = -1
        # ithetas same shape as ius, for each (x,y,theta) provides the index of theta
        ithetas = np.tile(np.arange(angles.shape[0]), (x.shape[0],y.shape[0],_z.shape[0],1)).astype(np.int)
        _ius = np.tile(ius, (1,1,_z.shape[0],1)).astype(np.int)

        V = _proj_data[ithetas,_ius,ivs]
        V_tot[:,:,i:i+buffer_size] = np.sum(V, axis=-1)

    return V_tot


from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import numpy as np

from odl.discr import DiscretizedSpace
from odl.tomo.geometry import Geometry

def wfbp(recon_space, geometry, proj_data):
    """
    First implementation of WFBP - WIP does not consider z!
    """
    angles = geometry.angles

    V = np.empty(recon_space.shape + angles.shape, dtype='float32', order='C')

    for ix,iy,iz in np.ndindex(recon_space.shape):
        x,y,z = recon_space.grid[ix,iy,iz]

        for itheta in range(angles.shape[0]):
            theta = angles[itheta]
            # clockwise rotation of theta 
            u = x*np.cos(theta) + y*np.sin(theta)
            iu = geometry.det_partition.byaxis[0].index(u)
            V[ix,iy,iz, itheta] = np.sum(proj_data[itheta,iu,:])
            
    return np.sum(V, axis=-1)

def wfbp_theta(recon_space, geometry, proj_data):
    """
    Implementation of WFBP with vectorised computation on theta - WIP does not consider z!
    """
    angles = geometry.angles
    upart = geometry.det_partition.byaxis[0]
    u_min = upart.min_pt
    u_cell = upart.cell_sides

    V_tot = np.empty(recon_space.shape, dtype='float32', order='C')

    for ix,iy,iz in np.ndindex(recon_space.shape):
        x,y,z = recon_space.grid[ix,iy,iz]
        
        us =  x*np.cos(angles) + y*np.sin(angles)
        ius = (us - u_min) // u_cell
        ius = ius.astype(np.int)
        V_tot[ix,iy,iz] = np.sum(proj_data[np.arange(angles.shape[0]),ius,:])
    return V_tot

def wfbp_theta(recon_space, geometry, proj_data):
    """
    Implementation of WFBP with vectorised computation on angles - WIP does not consider z!
    """
    angles = geometry.angles
    upart = geometry.det_partition.byaxis[0]
    u_min = upart.min_pt
    u_cell = upart.cell_sides

    V_tot = np.empty(recon_space.shape, dtype='float32', order='C')

    for ix,iy,iz in np.ndindex(recon_space.shape):
        x,y,z = recon_space.grid[ix,iy,iz]
        
        us =  x*np.cos(angles) + y*np.sin(angles)
        ius = (us - u_min) // u_cell
        ius = ius.astype(np.int)
        V_tot[ix,iy,iz] = np.sum(proj_data[np.arange(angles.shape[0]),ius,:])
    return V_tot

def wfbp_full(recon_space, geometry, proj_data):
    """
    Implementation of WFBP fully vectorised - WIP does not consider z!
    This is just an exercise!
    """
    angles = geometry.angles
    upart = geometry.det_partition.byaxis[0]
    u_min = upart.min_pt
    u_cell = upart.cell_sides

    V_tot = np.empty(recon_space.shape, dtype='float32', order='C')

    x, y, z = recon_space.grid.coord_vectors
    # shapes: (x, 1, angles) + (1, y, angles) -> (x,y,angles)
    # given x,y,theta compute u = x*np.cos(theta) + y*np.sin(theta)
    us = (np.outer(x, np.cos(angles)).reshape((x.shape[0],1,angles.shape[0])) +
          np.outer(y, np.sin(angles)).reshape((1,y.shape[0],angles.shape[0])))
    # get grid indices of us
    ius = (us - u_min) // u_cell
    ius = ius.astype(np.int)
    # ithetas same shape as ius, for each (x,y,theta) provides the index of theta
    ithetas = np.tile(np.arange(angles.shape[0]), (x.shape[0],y.shape[0],1))
    single_slice = np.sum(proj_data[ithetas,ius,:], axis=(-2,-1))
    # TODO: z is not considered!!! 
    V_tot = np.tile(np.expand_dims(single_slice,-1), (1, V_tot.shape[2]))

    return V_tot
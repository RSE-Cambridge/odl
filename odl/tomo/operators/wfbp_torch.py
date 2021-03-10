
from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import numpy as np
import torch

from odl.discr import DiscretizedSpace
from odl.tomo.geometry import Geometry

def wfbp_full_torch(recon_space, geometry, proj_data, dtype=torch.float32):
    """
    Torch implementation of WFBP - TODO: improve performances
    TODO: COPYING THE OUTPUT BACK TO CPU IS TOO SLOW

    Tested on det: (736,64) angles: 500 volume: (512,512,96) ~ 3.7 s (3.4 s float16)

    Note: uncomment code to change V memory layout (x, y, z, angles) -> (angles, x, y, z)
    Tested on det: (736,64) angles: 500 volume: (512,512,96) ~ 3.5 s (3.2 s float16)
    """

    cuda = torch.device('cuda')
    _proj_data = np.pad(proj_data, [(0,0),(0,0),(0,1)], mode='constant')    
    _proj_data = torch.tensor(_proj_data, dtype=dtype, device=cuda)
    angles = torch.tensor(geometry.angles, dtype=dtype, device=cuda)
    
    zs_src = angles * geometry.pitch / (2 * np.pi)
    zs_src = zs_src.reshape((1,1,-1))
#     zs_src = zs_src.reshape((-1,1,1))
    R = geometry.det_radius + geometry.src_radius

    u_min, v_min = geometry.det_partition.min_pt
    u_max, v_max = geometry.det_partition.max_pt
    u_cell, v_cell = geometry.det_partition.cell_sides

    x, y, z = recon_space.grid.coord_vectors
    x = torch.tensor(x, dtype=dtype, device=cuda)
    y = torch.tensor(y, dtype=dtype, device=cuda)
    z = torch.tensor(z, dtype=dtype, device=cuda)


    V_tot = torch.empty(recon_space.shape, dtype=dtype, device=cuda)
    V = torch.empty((x.shape[0],y.shape[0],angles.shape[0]), dtype=dtype, device=cuda)
    # V = torch.empty((angles.shape[0],x.shape[0],y.shape[0]), dtype=dtype, device=cuda)

    ithetas = np.tile(np.arange(angles.shape[0]), (x.shape[0],y.shape[0],1))
#     shape = (angles.shape[0],x.shape[0],y.shape[0])
#     ithetas = np.repeat(np.arange(shape[0]), np.prod(shape[1:])).reshape(shape)
    ithetas = torch.tensor(ithetas, dtype=torch.int64, device=cuda)
    
    
    # shapes: (x, 1, angles) + (1, y, angles) -> (x,y,angles)
    # given x,y,theta compute u = x*np.cos(theta) + y*np.sin(theta)
    us = (torch.outer(x, torch.cos(angles)).reshape((x.shape[0],1,angles.shape[0])) +
          torch.outer(y, torch.sin(angles)).reshape((1,y.shape[0],angles.shape[0])))
    ls = (torch.outer(x, -torch.sin(angles)).reshape((x.shape[0],1,angles.shape[0])) +
          torch.outer(y, torch.cos(angles)).reshape((1,y.shape[0],angles.shape[0])))
    
#     us = (torch.outer(torch.cos(angles), x).reshape((angles.shape[0],x.shape[0],1)) +
#           torch.outer(torch.sin(angles), y).reshape((angles.shape[0],1,y.shape[0])))
#     ls = (torch.outer(-torch.sin(angles), x).reshape((angles.shape[0],x.shape[0],1)) +
#           torch.outer(torch.cos(angles), y).reshape((angles.shape[0],1,y.shape[0])))
    
    # get grid indices
    ius = ((us - u_min) // u_cell).to(torch.int64)
    del us, x, y

    vs = (z[0] - zs_src) * R / (ls + geometry.src_radius)
    _ivs = ((vs - v_min) / v_cell)
    _ivs_delta = (z[1] - z[0]) * R / (ls + geometry.src_radius) / v_cell
    del vs, ls
    
    # split z to avoid memory problems:
    for i,_z in enumerate(z):

        ivs = _ivs.to(torch.int64)
        ivs = torch.where(ivs > 0, ivs, -1)
        ivs = torch.where(ivs < proj_data.shape[2], ivs, -1)


        V = _proj_data[ithetas,ius,ivs]
        
        V_tot[:,:,i] = torch.sum(V, axis=-1)
#         V_tot[:,:,i] = torch.sum(V, axis=0)
        
        _ivs += _ivs_delta

    return V_tot.cpu()

from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import numpy as np

from odl.discr import DiscretizedSpace
from odl.tomo.geometry import Geometry

def wfbp_theta_torch(recon_space, geometry, proj_data):
    """
    First Torch implementation - TODO: remove it
    """
    cuda = torch.device('cuda')
    
    
    proj_data = torch.tensor(proj_data, device=cuda)
    angles = torch.tensor(geometry.angles, device=cuda)
    zs_src = angles * geometry.pitch / (2 * np.pi)
    R = geometry.det_radius + geometry.src_radius

    u_min, v_min = geometry.det_partition.min_pt
    u_max, v_max = geometry.det_partition.max_pt
    u_cell, v_cell = geometry.det_partition.cell_sides


    V_tot = torch.empty(recon_space.shape, dtype=torch.float32, device=cuda)

    # Other indices
    us = torch.empty(angles.shape, dtype=torch.float32, device=cuda)
    vs = torch.empty(angles.shape, dtype=torch.float32, device=cuda)
    ls = torch.empty(angles.shape, dtype=torch.float32, device=cuda)

    for ix,iy,iz in np.ndindex(recon_space.shape):
        x,y,z = recon_space.grid[ix,iy,iz]
        # get u,v for rays throug the voxel at each angle
        # clockwise rotation by angles
        us = x*torch.cos(angles) + y*torch.sin(angles)
        ls = -x*torch.sin(angles) + y*torch.cos(angles)
        vs = (z - zs_src) * R / (ls + geometry.src_radius)

        # add mask for rays hitting the detector
        mask  = vs > v_min
        mask *= vs < v_max

        us = us[mask]
        vs = vs[mask]

        ius = ((us - u_min) // u_cell).to(torch.int64)
        ivs = ((vs - v_min) // v_cell).to(torch.int64)

        ithetas = torch.arange(angles.shape[0], dtype=torch.long, device=cuda)[mask]

        V_tot[ix,iy,iz] = torch.sum(proj_data[ithetas,ius,ivs])
    return V_tot.cpu()

def wfbp_full_torch(recon_space, geometry, proj_data):
    """
    Torch implementation of WFBP - TODO: improve performances

    Tested on det: (736,64) angles: 500 volume: (512,512,96) ~ 6 s
    """

    cuda = torch.device('cuda')
    _proj_data = np.pad(proj_data, [(0,0),(0,0),(0,1)], mode='constant')    
    proj_data = torch.tensor(_proj_data, device=cuda)
    angles = torch.tensor(geometry.angles, device=cuda)
    
    zs_src = angles * geometry.pitch / (2 * np.pi)
    zs_src = zs_src.reshape((1,1,1,-1))
    R = geometry.det_radius + geometry.src_radius

    u_min, v_min = geometry.det_partition.min_pt
    u_max, v_max = geometry.det_partition.max_pt
    u_cell, v_cell = geometry.det_partition.cell_sides

    V_tot = torch.empty(recon_space.shape, dtype=torch.float32, device=cuda)

    # Other indices
    x, y, z = recon_space.grid.coord_vectors
    x = torch.tensor(x, device=cuda)
    y = torch.tensor(y, device=cuda)
    z = torch.tensor(z, device=cuda)   
        
    ithetas = np.tile(np.arange(angles.shape[0]), (x.shape[0],y.shape[0],1))
    ithetas = torch.tensor(ithetas, dtype=torch.int64, device=cuda)

    # shapes: (x, 1, angles) + (1, y, angles) -> (x,y,angles)
    # given x,y,theta compute u = x*np.cos(theta) + y*np.sin(theta)
    us = (torch.outer(x, torch.cos(angles)).reshape((x.shape[0],1,angles.shape[0])) +
          torch.outer(y, torch.sin(angles)).reshape((1,y.shape[0],angles.shape[0])))
    ls = (torch.outer(x, -torch.sin(angles)).reshape((x.shape[0],1,angles.shape[0])) +
          torch.outer(y, torch.cos(angles)).reshape((1,y.shape[0],angles.shape[0])))

    # get grid indices
    ius = ((us - u_min) // u_cell).to(torch.int64)

    # split z to avoid memory problems:
    for i,_z in enumerate(z):

        vs = (_z - zs_src) * R / (ls + geometry.src_radius)
        
        mask  = vs < v_min
        mask += vs > v_max

    
        ivs = ((vs - v_min) // v_cell).to(torch.int64)
        

        # Deal with ivs out of range
        ivs[mask] = -1


        V = proj_data[ithetas,ius,ivs]

        
        V_tot[:,:,i] = torch.sum(V, axis=-1)

    return V_tot.cpu()

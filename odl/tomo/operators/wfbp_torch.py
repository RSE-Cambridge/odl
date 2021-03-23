
from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F


from odl.discr import DiscretizedSpace
from odl.tomo.geometry import Geometry

def wfbp_torch_angles(recon_space, geometry, proj_data,
                dtype=torch.float32, in_mem=False, t_chunk=100):
    """
    Torch implementation of Siemens BP - Angles chunked, loop over angles

    Notes:
    * Tested on det: (736,64) angles: 500 volume: (512,512,96)
      ~ 1.39 s (~1.15 s float16)
    * indices memory layout: (angles, x, y, z)
    * Return V_tot - memory layout: (x, y, z)
    """
    cuda = torch.device('cuda')

    V_tot = torch.zeros(recon_space.shape, dtype=dtype, device=cuda)

    R = geometry.det_radius + geometry.src_radius

    u_min, v_min = geometry.det_partition.min_pt
    u_max, v_max = geometry.det_partition.max_pt
    u_cell, v_cell = geometry.det_partition.cell_sides
    
    for i_angle in range(0, geometry.angles.shape[0], t_chunk):
        # Chuncking projection data speed up computation for high humber of angles 
        if isinstance(proj_data, torch.Tensor):
            _proj_data = proj_data[i_angle:i_angle+t_chunk].cuda()
            _proj_data = F.pad(_proj_data, (0,1), mode='constant', value=0)
        else:
            _proj_data = np.pad(proj_data[i_angle:i_angle+t_chunk], [(0,0),(0,0),(0,1)], mode='constant')
            _proj_data = torch.tensor(_proj_data, dtype=dtype, device=cuda)

        x, y, z = recon_space.grid.coord_vectors
        x = torch.tensor(x, dtype=dtype, device=cuda)
        y = torch.tensor(y, dtype=dtype, device=cuda)
        z = torch.tensor(z, dtype=dtype, device=cuda)
        z = z.reshape((1, 1, -1))

        angles = torch.tensor(geometry.angles[i_angle:i_angle+t_chunk],
                              dtype=dtype, device=cuda)

        zs_src = angles * geometry.pitch / (2 * np.pi) + geometry.offset_along_axis

        # shapes: (angles, x, 1, 1) + (angles, 1, y, 1) -> (angles, x, y, 1)
        us = (torch.outer(torch.cos(angles), x).reshape((angles.shape[0], x.shape[0], 1, 1)) +
            torch.outer(torch.sin(angles), y).reshape((angles.shape[0], 1, y.shape[0], 1)))
        ls = (torch.outer(-torch.sin(angles), x).reshape((angles.shape[0], x.shape[0], 1, 1)) +
            torch.outer(torch.cos(angles), y).reshape((angles.shape[0], 1, y.shape[0], 1)))
        ls += geometry.src_radius

        _ius = ((us - u_min) // u_cell).to(torch.int64)
        del us, x, y

        i_v_min = torch.tensor(0, dtype=dtype)
        i_v_max = torch.tensor(proj_data.shape[2], dtype=dtype)
        i_repalce = torch.tensor(-1, dtype=dtype)

        for i in range(angles.shape[0]):
            ius = _ius[i]
            vs = (z - zs_src[i]) * R / ls[i]

            _ivs = (vs - v_min) / v_cell
            mask = _ivs < i_v_min
            mask += _ivs >= i_v_max
            _ivs.masked_fill_( mask , i_repalce)
            ivs = _ivs.to(torch.int64)

            V_tot += _proj_data[i,ius,ivs]

    if in_mem:
        return V_tot
    else:
        return V_tot.cpu()

def wfbp_torch_z_tuv(recon_space, geometry, proj_data,
                    dtype=torch.float32, in_mem=False, t_chunk=100):
    """
    Torch implementation of Siemens BP - Angles chunked, loop over z

    Notes:
    * indices memory layout: (angles, x, y)
    * Return V_tot - memory layout: (x, y, z)
    """

    cuda = torch.device('cuda')

    V_tot = torch.zeros(recon_space.shape, dtype=dtype, device=cuda)

    R = geometry.det_radius + geometry.src_radius

    u_min, v_min = geometry.det_partition.min_pt
    u_max, v_max = geometry.det_partition.max_pt
    u_cell, v_cell = geometry.det_partition.cell_sides

    i_v_min = torch.tensor(0, dtype=dtype)
    i_v_max = torch.tensor(proj_data.shape[2], dtype=dtype)
    i_repalce = torch.tensor(-1, dtype=dtype)

    for i_angle in range(0, geometry.angles.shape[0], t_chunk):
        # Why this has to be inside the for loop? If not:
        # UnboundLocalError: local variable 'x' referenced before assignment
        x, y, z = recon_space.grid.coord_vectors
        x = torch.tensor(x, dtype=dtype, device=cuda)
        y = torch.tensor(y, dtype=dtype, device=cuda)
        z = torch.tensor(z, dtype=dtype, device=cuda)

        angles = torch.tensor(geometry.angles[i_angle:i_angle+t_chunk], dtype=dtype, device=cuda)
        V = torch.empty((x.shape[0],y.shape[0],angles.shape[0]), dtype=dtype, device=cuda)

        zs_src = angles * geometry.pitch / (2 * np.pi) + geometry.offset_along_axis
        zs_src = zs_src.reshape((-1,1,1))

        if isinstance(proj_data, torch.Tensor):
            _proj_data = proj_data[i_angle:i_angle+t_chunk].cuda()
            _proj_data = F.pad(_proj_data, (0,1), mode='constant', value=0)
        else:
            _proj_data = np.pad(proj_data[i_angle:i_angle+t_chunk],
                                [(0,0),(0,0),(0,1)], mode='constant')
            _proj_data = torch.tensor(_proj_data, dtype=dtype, device=cuda)

        # ithetas = np.tile(np.arange(angles.shape[0]), (x.shape[0],y.shape[0],1))
        ithetas = np.arange(angles.shape[0]).reshape((-1,1,1))
        ithetas = torch.tensor(ithetas, dtype=torch.int64, device=cuda)


        # shapes: (angles, x, 1) + (angles, 1, y) -> (angles, x, y)
        us = (torch.outer(torch.cos(angles), x).reshape((angles.shape[0],x.shape[0],1)) +
            torch.outer(torch.sin(angles), y).reshape((angles.shape[0],1,y.shape[0])))
        ls = (torch.outer(-torch.sin(angles), x).reshape((angles.shape[0],x.shape[0],1)) +
            torch.outer(torch.cos(angles), y).reshape((angles.shape[0],1,y.shape[0])))
        ls += geometry.src_radius
        # get grid indices
        ius = ((us - u_min) // u_cell).to(torch.int64)
        del us, x, y

        ratio = R / ls
        del ls


        # split z to avoid memory problems:
        for i,_z in enumerate(z):
            vs = (_z - zs_src) * ratio
            _ivs = ((vs - v_min) / v_cell)
            mask = _ivs < i_v_min
            mask += _ivs >= i_v_max
            _ivs.masked_fill_( mask , i_repalce)
            ivs = _ivs.to(torch.int64)

            V = _proj_data[ithetas,ius,ivs]
            V_tot[:,:,i] += torch.sum(V, axis=0)

    if in_mem:
        return V_tot
    else:
        return V_tot.cpu()

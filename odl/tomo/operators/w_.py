
from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from odl.discr import DiscretizedSpace
from odl.tomo.geometry import Geometry

def w_a_last(recon_space, geometry, proj_data, sync=False,
                dtype=torch.float32, in_mem=False):
    '''_zxy'''
    cuda = torch.device('cuda')

    if isinstance(proj_data, torch.Tensor):
        _proj_data = proj_data.cuda()
        _proj_data = F.pad(_proj_data, (0,1), mode='constant', value=0)
    else:
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

    shape = (z.shape[0], x.shape[0], y.shape[0])
    V_tot = torch.empty(shape, dtype=dtype, device=cuda)
    V = torch.empty((x.shape[0],y.shape[0],angles.shape[0]), dtype=dtype, device=cuda)

        
    ithetas = np.arange(angles.shape[0]).reshape((1,1,-1))
    ithetas = torch.tensor(ithetas, dtype=torch.int64, device=cuda)
    
    
    # shapes: (x, 1, angles) + (1, y, angles) -> (x,y,angles)
    # given x,y,theta compute u = x*np.cos(theta) + y*np.sin(theta)
    us = (torch.outer(x, torch.cos(angles)).reshape((x.shape[0],1,angles.shape[0])) +
          torch.outer(y, torch.sin(angles)).reshape((1,y.shape[0],angles.shape[0])))
    ls = (torch.outer(x, -torch.sin(angles)).reshape((x.shape[0],1,angles.shape[0])) +
          torch.outer(y, torch.cos(angles)).reshape((1,y.shape[0],angles.shape[0])))
    
    # print(us.dtype, ls.dtype, x.dtype, y.dtype, z.dtype)
    # get grid indices
    ius = ((us - u_min) // u_cell).to(torch.int64)
    del us, x, y

    vs = (z[0] - zs_src) * R / (ls + geometry.src_radius)
    _ivs = ((vs - v_min) / v_cell)
    _ivs_delta = (z[1] - z[0]) * R / (ls + geometry.src_radius) / v_cell
    del vs, ls
    

    i_v_min = torch.tensor(0, dtype=dtype)
    i_v_max = torch.tensor(proj_data.shape[2], dtype=dtype)
    i_repalce = torch.tensor(-1, dtype=dtype)

    # split z to avoid memory problems:
    for i,_z in enumerate(z):

        mask = _ivs < i_v_min
        mask += _ivs >= i_v_max
        _ivs.masked_fill_( mask , i_repalce)

        ivs = _ivs.to(torch.int64)

        if sync:
            torch.cuda.synchronize()


        V = _proj_data[ithetas,ius,ivs]

        if sync:
            torch.cuda.synchronize()

        V_tot[i] = torch.sum(V, axis=-1)

        if sync:
            torch.cuda.synchronize()

        _ivs += _ivs_delta
    
    # print(V.dtype, V_tot.dtype, ivs.dtype, _ivs.dtype, _ivs_delta.dtype)

    if in_mem:
        return V_tot
    else:
        return V_tot.cpu()


def w_a_first(recon_space, geometry, proj_data, sync=False,
                dtype=torch.float32, in_mem=False):
 
    cuda = torch.device('cuda')

    if isinstance(proj_data, torch.Tensor):
        _proj_data = proj_data.cuda()
        _proj_data = F.pad(_proj_data, (0,1), mode='constant', value=0)
    else:
        _proj_data = np.pad(proj_data, [(0,0),(0,0),(0,1)], mode='constant')
        _proj_data = torch.tensor(_proj_data, dtype=dtype, device=cuda)

    angles = torch.tensor(geometry.angles, dtype=dtype, device=cuda)
    
    zs_src = angles * geometry.pitch / (2 * np.pi)
    zs_src = zs_src.reshape((-1,1,1))
    R = geometry.det_radius + geometry.src_radius

    u_min, v_min = geometry.det_partition.min_pt
    u_max, v_max = geometry.det_partition.max_pt
    u_cell, v_cell = geometry.det_partition.cell_sides

    V_tot = torch.empty(recon_space.shape, dtype=dtype, device=cuda)

    x, y, z = recon_space.grid.coord_vectors
    x = torch.tensor(x, dtype=dtype, device=cuda)
    y = torch.tensor(y, dtype=dtype, device=cuda)
    z = torch.tensor(z, dtype=dtype, device=cuda)
    
    V = torch.empty((angles.shape[0],x.shape[0],y.shape[0]), dtype=dtype, device=cuda)

    ithetas = np.arange(angles.shape[0]).reshape((-1,1,1))
    ithetas = torch.tensor(ithetas, dtype=torch.int64, device=cuda)
    
    
    # shapes: (angles, x, 1 ) + (angles, 1, y) -> (angles, x, y)
    # given x,y,theta compute u = x*np.cos(theta) + y*np.sin(theta)
    
    us = (torch.outer(torch.cos(angles), x).reshape((angles.shape[0],x.shape[0],1)) +
          torch.outer(torch.sin(angles), y).reshape((angles.shape[0],1,y.shape[0])))
    ls = (torch.outer(-torch.sin(angles), x).reshape((angles.shape[0],x.shape[0],1)) +
          torch.outer(torch.cos(angles), y).reshape((angles.shape[0],1,y.shape[0])))
    
    # get grid indices
    ius = ((us - u_min) // u_cell).to(torch.int64)
    del us, x, y

    vs = (z[0] - zs_src) * R / (ls + geometry.src_radius)
    _ivs = ((vs - v_min) / v_cell)
    _ivs_delta = (z[1] - z[0]) * R / (ls + geometry.src_radius) / v_cell
    del vs, ls
    
    i_v_min = torch.tensor(0, dtype=dtype)
    i_v_max = torch.tensor(proj_data.shape[2], dtype=dtype)
    i_repalce = torch.tensor(-1, dtype=dtype)

    # split z to avoid memory problems:
    for i,_z in enumerate(z):

        mask = _ivs < i_v_min
        mask += _ivs >= i_v_max
        _ivs.masked_fill_( mask , i_repalce)

        ivs = _ivs.to(torch.int64)

        if sync:
            torch.cuda.synchronize()

        V = _proj_data[ithetas,ius,ivs]

        if sync:
            torch.cuda.synchronize()
        
        V_tot[:,:,i] = torch.sum(V, axis=0)

        if sync:
            torch.cuda.synchronize()
        
        _ivs += _ivs_delta

    if in_mem:
        return V_tot
    else:
        return V_tot.cpu()


def w_angles(recon_space, geometry, proj_data,
                dtype=torch.float32, in_mem=False):

    cuda = torch.device('cuda')

    if isinstance(proj_data, torch.Tensor):
        _proj_data = proj_data.cuda()
        _proj_data = F.pad(_proj_data, (0,1), mode='constant', value=0)
    else:
        _proj_data = np.pad(proj_data, [(0,0),(0,0),(0,1)], mode='constant')
        _proj_data = torch.tensor(_proj_data, dtype=dtype, device=cuda)

    angles = torch.tensor(geometry.angles, dtype=dtype, device=cuda)
    
    zs_src = angles * geometry.pitch / (2 * np.pi)

    R = geometry.det_radius + geometry.src_radius

    u_min, v_min = geometry.det_partition.min_pt
    u_max, v_max = geometry.det_partition.max_pt
    u_cell, v_cell = geometry.det_partition.cell_sides

    x, y, z = recon_space.grid.coord_vectors
    x = torch.tensor(x, dtype=dtype, device=cuda)
    y = torch.tensor(y, dtype=dtype, device=cuda)
    z = torch.tensor(z, dtype=dtype, device=cuda)
    z = z.reshape((1, 1, -1))

    V_tot = torch.zeros(recon_space.shape, dtype=dtype, device=cuda)

    # shapes: (angles, x, 1, 1) + (angles, 1, y, 1) -> (angles, x, y, 1)
    us = (torch.outer(torch.cos(angles), x).reshape((angles.shape[0], x.shape[0], 1, 1)) +
          torch.outer(torch.sin(angles), y).reshape((angles.shape[0], 1, y.shape[0], 1)))
    ls = (torch.outer(-torch.sin(angles), x).reshape((angles.shape[0], x.shape[0], 1, 1)) +
          torch.outer(torch.cos(angles), y).reshape((angles.shape[0], 1, y.shape[0], 1)))
    ls += geometry.src_radius

    # get grid indices
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




def w_utv(recon_space, geometry, proj_data, sync=False,
                dtype=torch.float32, in_mem=False):
    cuda = torch.device('cuda')

    if isinstance(proj_data, torch.Tensor):
        _proj_data = proj_data.cuda()
        _proj_data = F.pad(_proj_data, (0,1), mode='constant', value=0)
    else:
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

    V_tot = torch.empty(z.shape[0], x.shape[0], y.shape[0], dtype=dtype, device=cuda)
    V = torch.empty((x.shape[0],y.shape[0],angles.shape[0]), dtype=dtype, device=cuda)

        
    ithetas = np.arange(angles.shape[0]).reshape((1,1,-1))
    ithetas = torch.tensor(ithetas, dtype=torch.int64, device=cuda)
    
    
    # shapes: (x, 1, angles) + (1, y, angles) -> (x,y,angles)
    # given x,y,theta compute u = x*np.cos(theta) + y*np.sin(theta)
    us = (torch.outer(x, torch.cos(angles)).reshape((x.shape[0],1,angles.shape[0])) +
          torch.outer(y, torch.sin(angles)).reshape((1,y.shape[0],angles.shape[0])))
    ls = (torch.outer(x, -torch.sin(angles)).reshape((x.shape[0],1,angles.shape[0])) +
          torch.outer(y, torch.cos(angles)).reshape((1,y.shape[0],angles.shape[0])))
    
    # print(us.dtype, ls.dtype, x.dtype, y.dtype, z.dtype)
    # get grid indices
    ius = ((us - u_min) // u_cell).to(torch.int64)
    del us, x, y

    vs = (z[0] - zs_src) * R / (ls + geometry.src_radius)
    _ivs = ((vs - v_min) / v_cell)
    _ivs_delta = (z[1] - z[0]) * R / (ls + geometry.src_radius) / v_cell
    del vs, ls
    

    i_v_min = torch.tensor(0, dtype=dtype)
    i_v_max = torch.tensor(proj_data.shape[2], dtype=dtype)
    i_repalce = torch.tensor(-1, dtype=dtype)

    # split z to avoid memory problems:
    for i,_z in enumerate(z):

        mask = _ivs < i_v_min
        mask += _ivs >= i_v_max
        _ivs.masked_fill_( mask , i_repalce)

        ivs = _ivs.to(torch.int64)

        if sync:
            torch.cuda.synchronize()


        V = _proj_data[ius,ithetas,ivs]

        if sync:
            torch.cuda.synchronize()

        V_tot[i] = torch.sum(V, axis=-1)

        if sync:
            torch.cuda.synchronize()

        _ivs += _ivs_delta
    
    # print(V.dtype, V_tot.dtype, ivs.dtype, _ivs.dtype, _ivs_delta.dtype)

    if in_mem:
        return V_tot
    else:
        return V_tot.cpu()

def w_vtu(recon_space, geometry, proj_data, sync=False,
                dtype=torch.float32, in_mem=False):
    cuda = torch.device('cuda')

    if isinstance(proj_data, torch.Tensor):
        _proj_data = proj_data.cuda()
        _proj_data = F.pad(_proj_data, (0,1,0,0,0,0), mode='constant', value=0)
    else:
        _proj_data = np.pad(proj_data, [(0,1),(0,0),(0,0)], mode='constant')
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

    V_tot = torch.empty(z.shape[0], x.shape[0], y.shape[0], dtype=dtype, device=cuda)
    V = torch.empty((x.shape[0],y.shape[0],angles.shape[0]), dtype=dtype, device=cuda)

    ithetas = np.arange(angles.shape[0]).reshape((1,1,-1))
    ithetas = torch.tensor(ithetas, dtype=torch.int64, device=cuda)
    
    
    # shapes: (x, 1, angles) + (1, y, angles) -> (x,y,angles)
    # given x,y,theta compute u = x*np.cos(theta) + y*np.sin(theta)
    us = (torch.outer(x, torch.cos(angles)).reshape((x.shape[0],1,angles.shape[0])) +
          torch.outer(y, torch.sin(angles)).reshape((1,y.shape[0],angles.shape[0])))
    ls = (torch.outer(x, -torch.sin(angles)).reshape((x.shape[0],1,angles.shape[0])) +
          torch.outer(y, torch.cos(angles)).reshape((1,y.shape[0],angles.shape[0])))
    
    # print(us.dtype, ls.dtype, x.dtype, y.dtype, z.dtype)
    # get grid indices
    ius = ((us - u_min) // u_cell).to(torch.int64)
    del us, x, y

    vs = (z[0] - zs_src) * R / (ls + geometry.src_radius)
    _ivs = ((vs - v_min) / v_cell)
    _ivs_delta = (z[1] - z[0]) * R / (ls + geometry.src_radius) / v_cell
    del vs, ls
    

    i_v_min = torch.tensor(0, dtype=dtype)
    i_v_max = torch.tensor(proj_data.shape[0], dtype=dtype)
    i_repalce = torch.tensor(-1, dtype=dtype)

    # split z to avoid memory problems:
    for i,_z in enumerate(z):

        mask = _ivs < i_v_min
        mask += _ivs >= i_v_max
        _ivs.masked_fill_( mask , i_repalce)

        ivs = _ivs.to(torch.int64)

        if sync:
            torch.cuda.synchronize()



        V = _proj_data[ivs,ithetas,ius]

        if sync:
            torch.cuda.synchronize()

        V_tot[i] = torch.sum(V, axis=-1)

        if sync:
            torch.cuda.synchronize()

        _ivs += _ivs_delta
    
    # print(V.dtype, V_tot.dtype, ivs.dtype, _ivs.dtype, _ivs_delta.dtype)

    if in_mem:
        return V_tot
    else:
        return V_tot.cpu()

def w_angles_vtu(recon_space, geometry, proj_data,
                dtype=torch.float32, in_mem=False):

    cuda = torch.device('cuda')

    if isinstance(proj_data, torch.Tensor):
        _proj_data = proj_data.cuda()
        _proj_data = F.pad(_proj_data, (0,1,0,0,0,0), mode='constant', value=0)
    else:
        _proj_data = np.pad(proj_data, [(0,1),(0,0),(0,0)], mode='constant')
        _proj_data = torch.tensor(_proj_data, dtype=dtype, device=cuda)

    angles = torch.tensor(geometry.angles, dtype=dtype, device=cuda)
  
    zs_src = angles * geometry.pitch / (2 * np.pi)

    R = geometry.det_radius + geometry.src_radius

    u_min, v_min = geometry.det_partition.min_pt
    u_max, v_max = geometry.det_partition.max_pt
    u_cell, v_cell = geometry.det_partition.cell_sides

    x, y, z = recon_space.grid.coord_vectors
    x = torch.tensor(x, dtype=dtype, device=cuda)
    y = torch.tensor(y, dtype=dtype, device=cuda)
    z = torch.tensor(z, dtype=dtype, device=cuda)
    z = z.reshape((1, 1, -1))

    V_tot = torch.zeros(recon_space.shape, dtype=dtype, device=cuda)

    # shapes: (angles, x, 1, 1) + (angles, 1, y, 1) -> (angles, x, y, 1)
    us = (torch.outer(torch.cos(angles), x).reshape((angles.shape[0], x.shape[0], 1, 1)) +
          torch.outer(torch.sin(angles), y).reshape((angles.shape[0], 1, y.shape[0], 1)))
    ls = (torch.outer(-torch.sin(angles), x).reshape((angles.shape[0], x.shape[0], 1, 1)) +
          torch.outer(torch.cos(angles), y).reshape((angles.shape[0], 1, y.shape[0], 1)))
    ls += geometry.src_radius

    # get grid indices
    _ius = ((us - u_min) // u_cell).to(torch.int64)
    del us, x, y

    i_v_min = torch.tensor(0, dtype=dtype)
    i_v_max = torch.tensor(proj_data.shape[0], dtype=dtype)
    i_repalce = torch.tensor(-1, dtype=dtype)

    for i in range(angles.shape[0]):
        ius = _ius[i]
        vs = (z - zs_src[i]) * R / ls[i]

        _ivs = (vs - v_min) / v_cell
        mask = _ivs < i_v_min
        mask += _ivs >= i_v_max
        _ivs.masked_fill_( mask , i_repalce)
        ivs = _ivs.to(torch.int64)

        V_tot += _proj_data[ivs,i,ius]

    if in_mem:
        return V_tot
    else:
        return V_tot.cpu()

def w_angles_tvu(recon_space, geometry, proj_data,
                dtype=torch.float32, in_mem=False):

    cuda = torch.device('cuda')

    if isinstance(proj_data, torch.Tensor):
        _proj_data = proj_data.cuda()
        _proj_data = F.pad(_proj_data, (0,1,0,0), mode='constant', value=0)
    else:
        _proj_data = np.pad(proj_data, [(0,0),(0,1),(0,0)], mode='constant')
        _proj_data = torch.tensor(_proj_data, dtype=dtype, device=cuda)

    angles = torch.tensor(geometry.angles, dtype=dtype, device=cuda)
    
    zs_src = angles * geometry.pitch / (2 * np.pi)

    R = geometry.det_radius + geometry.src_radius

    u_min, v_min = geometry.det_partition.min_pt
    u_max, v_max = geometry.det_partition.max_pt
    u_cell, v_cell = geometry.det_partition.cell_sides

    x, y, z = recon_space.grid.coord_vectors
    x = torch.tensor(x, dtype=dtype, device=cuda)
    y = torch.tensor(y, dtype=dtype, device=cuda)
    z = torch.tensor(z, dtype=dtype, device=cuda)
    z = z.reshape((1, 1, -1))

    V_tot = torch.zeros(recon_space.shape, dtype=dtype, device=cuda)

    # shapes: (angles, x, 1, 1) + (angles, 1, y, 1) -> (angles, x, y, 1)
    us = (torch.outer(torch.cos(angles), x).reshape((angles.shape[0], x.shape[0], 1, 1)) +
          torch.outer(torch.sin(angles), y).reshape((angles.shape[0], 1, y.shape[0], 1)))
    ls = (torch.outer(-torch.sin(angles), x).reshape((angles.shape[0], x.shape[0], 1, 1)) +
          torch.outer(torch.cos(angles), y).reshape((angles.shape[0], 1, y.shape[0], 1)))
    ls += geometry.src_radius

    # get grid indices
    _ius = ((us - u_min) // u_cell).to(torch.int64)
    del us, x, y

    i_v_min = torch.tensor(0, dtype=dtype)
    i_v_max = torch.tensor(proj_data.shape[1], dtype=dtype)
    i_repalce = torch.tensor(-1, dtype=dtype)

    for i in range(angles.shape[0]):
        ius = _ius[i]
        vs = (z - zs_src[i]) * R / ls[i]

        _ivs = (vs - v_min) / v_cell
        mask = _ivs < i_v_min
        mask += _ivs >= i_v_max
        _ivs.masked_fill_( mask , i_repalce)
        ivs = _ivs.to(torch.int64)

        V_tot += _proj_data[i,ivs,ius]

    if in_mem:
        return V_tot
    else:
        return V_tot.cpu()


def w_angles_tvu_zxy(recon_space, geometry, proj_data,
                dtype=torch.float32, in_mem=False):

    cuda = torch.device('cuda')

    if isinstance(proj_data, torch.Tensor):
        _proj_data = proj_data.cuda()
        _proj_data = F.pad(_proj_data, (0,1,0,0), mode='constant', value=0)
    else:
        _proj_data = np.pad(proj_data, [(0,0),(0,1),(0,0)], mode='constant')
        _proj_data = torch.tensor(_proj_data, dtype=dtype, device=cuda)

    angles = torch.tensor(geometry.angles, dtype=dtype, device=cuda)
    
    zs_src = angles * geometry.pitch / (2 * np.pi)

    R = geometry.det_radius + geometry.src_radius

    u_min, v_min = geometry.det_partition.min_pt
    u_max, v_max = geometry.det_partition.max_pt
    u_cell, v_cell = geometry.det_partition.cell_sides

    x, y, z = recon_space.grid.coord_vectors
    x = torch.tensor(x, dtype=dtype, device=cuda)
    y = torch.tensor(y, dtype=dtype, device=cuda)
    z = torch.tensor(z, dtype=dtype, device=cuda)
    z = z.reshape((-1, 1, 1))

    shape = (z.shape[0], x.shape[0], y.shape[0])
    V_tot = torch.zeros(shape, dtype=dtype, device=cuda)

    # shapes: (angles, x, 1, 1) + (angles, 1, y, 1) -> (angles, x, y, 1)
    us = (torch.outer(torch.cos(angles), x).reshape((angles.shape[0], 1, x.shape[0], 1)) +
          torch.outer(torch.sin(angles), y).reshape((angles.shape[0], 1, 1, y.shape[0])))
    ls = (torch.outer(-torch.sin(angles), x).reshape((angles.shape[0], 1, x.shape[0], 1)) +
          torch.outer(torch.cos(angles), y).reshape((angles.shape[0], 1, 1, y.shape[0])))
    ls += geometry.src_radius

    # get grid indices
    _ius = ((us - u_min) // u_cell).to(torch.int64)
    del us, x, y

    i_v_min = torch.tensor(0, dtype=dtype)
    i_v_max = torch.tensor(proj_data.shape[1], dtype=dtype)
    i_repalce = torch.tensor(-1, dtype=dtype)

    for i in range(angles.shape[0]):
        ius = _ius[i]
        # print(z.shape, zs_src[i].shape, ls[i].shape)
        vs = (z - zs_src[i]) * R / ls[i]

        _ivs = (vs - v_min) / v_cell
        mask = _ivs < i_v_min
        mask += _ivs >= i_v_max
        _ivs.masked_fill_( mask , i_repalce)
        ivs = _ivs.to(torch.int64)

        V_tot += _proj_data[i,ivs,ius]
    
    if in_mem:
        return V_tot
    else:
        return V_tot.cpu()
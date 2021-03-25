
from __future__ import absolute_import, division, print_function


import numpy as np
import torch
import torch.nn.functional as F

__all__ = ('siemens_bp',)

def siemens_bp(reco_space, geometry, proj_data,
               dtype=torch.float32, in_mem=False, angle_chunk=100):
    """
    Torch implementation of Siemens BP.
    Check Stierstorfer's 2004 paper: `
    <https://iopscience.iop.org/article/10.1088/0031-9155/49/11/007/meta>`_.

    For each voxel in the reconstruction space x, y, z and for each
    acquisition angle theta:
    - Get the parameters (u, v) in the detector local coordinates of
     the voxel projection at angle theta.
    - Compute the indices (iu, iv) of the corresponding detector cell
     (set to -1 if out of bound).
    Up to a weighting function on the detector rows and a normalisation
    factor (not implemented yet), the reconstruction value on a voxel
    is obtained summing the detecor acquisitions at (iu, iv) over all
    theta (equation 13 in the paper).

    Notes:
    * Tested on det: (736,64); angles: 500; volume: (512,512,96);
      ~ 1.39 s (~1.15 s float16).
    * indices memory layout: (angles, x, y, z).
    * The channel indices of a projection ``iu`` do not depend on z.
      That is why _ius has shape (angles, x, y, 1).
    * This is not the case for the row indices ``iv``. These are
      computed inside the for loop one angle a time.
    """
    cuda = torch.device('cuda')

    V_tot = torch.zeros(reco_space.shape, dtype=dtype, device=cuda)

    u_min, v_min = geometry.det_partition.min_pt
    u_max, v_max = geometry.det_partition.max_pt
    u_cell, v_cell = geometry.det_partition.cell_sides

    det_to_src = geometry.det_radius + geometry.src_radius
    # Chunk workload to speed up computation for high humber of angles
    for i_angle in range(0, geometry.angles.shape[0], angle_chunk):
        # Check if projection data is already a Torch tensor and load
        # it on GPU. Pad the last dimension of projection data with
        # zeros to deal with out of bound indices.
        if isinstance(proj_data, torch.Tensor):
            _proj_data = proj_data[i_angle:i_angle+angle_chunk].cuda()
            _proj_data = F.pad(_proj_data,
                               (0,1),
                               mode='constant',
                               value=0)
        else:
            _proj_data = np.pad(proj_data[i_angle:i_angle+angle_chunk],
                                [(0,0),(0,0),(0,1)],
                                mode='constant')
            _proj_data = torch.tensor(_proj_data, dtype=dtype, device=cuda)

        x, y, z = reco_space.grid.coord_vectors
        x = torch.tensor(x, dtype=dtype, device=cuda)
        y = torch.tensor(y, dtype=dtype, device=cuda)
        z = torch.tensor(z, dtype=dtype, device=cuda)
        z = z.reshape((1, 1, -1))

        # Load a chunk of angles
        angles = torch.tensor(geometry.angles[i_angle:i_angle+angle_chunk],
                              dtype=dtype, device=cuda)
        # Compute the z coordinates of the acquisition points
        zs_src = (angles * geometry.pitch / (2 * np.pi)
                  + geometry.offset_along_axis)

        # For each angle theta, compute (u, l) the coordinates on the
        # axial plane of the voxels after clockwise rotation by theta.
        # shapes: (angles, x, 1, 1) + (angles, 1, y, 1) -> (angles, x, y, 1)
        us = (torch.outer(torch.cos(angles), x).reshape((angles.shape[0], x.shape[0], 1, 1))
              + torch.outer(torch.sin(angles), y).reshape((angles.shape[0], 1, y.shape[0], 1)))
        ls = (torch.outer(-torch.sin(angles), x).reshape((angles.shape[0], x.shape[0], 1, 1))
              + torch.outer(torch.cos(angles), y).reshape((angles.shape[0], 1, y.shape[0], 1)))
        ls += geometry.src_radius
        # u = first detector parameter (channel direction) of voxels'
        #     projections at angle theta
        # l = distance between voxels and the source at angle theta on
        #     the axial plane

        # Get the channel indices of the projections
        _ius = ((us - u_min) // u_cell).to(torch.int64)
        del us, x, y

        i_v_min = torch.tensor(0, dtype=dtype)
        i_v_max = torch.tensor(proj_data.shape[2], dtype=dtype)
        i_repalce = torch.tensor(-1, dtype=dtype)

        # Loop on angles
        for i in range(angles.shape[0]):
            # Load the channel indices at angle theta
            ius = _ius[i]
            # Compute the second detector parameter (row direction) of
            # voxels' projections at angle theta
            vs = (z - zs_src[i]) * det_to_src / ls[i]
            # Get the channel indices of the projections
            _ivs = (vs - v_min) / v_cell
            # Replace out of bound indices with -1
            mask = _ivs < i_v_min
            mask += _ivs >= i_v_max
            _ivs.masked_fill_( mask , i_repalce)
            # Convert to long int
            ivs = _ivs.to(torch.int64)
            # Sum projection data over angles (eq. 13)
            V_tot += _proj_data[i,ius,ivs]

    if in_mem:
        return V_tot
    else:
        return V_tot.cpu()

"""Example using the Siemens BP on 3d tilted booklets geometry."""

import torch
import numpy as np
import odl

import matplotlib.pyplot as plt
from odl.util.testutils import timer
from odl.tomo.geometry import TiltedBookletsGeometry
from odl.tomo.backends import siemens_bp


# Common geometry parameters

n_angles = 500
reco_shape = (512,512,24)
det_shape = (736, 64)

print(f'Angles: {n_angles}')
print(f'Reconstruction space shape: {reco_shape}')
print(f'Detecor shape: {det_shape}')

apart = odl.uniform_partition(0,4*np.pi, n_angles)
dpart = odl.uniform_partition([-1, -0.1], [1, .1], (736, 64)) 

reco_space = odl.uniform_discr([-0.25, -0.25, 0], [0.25, 0.25, 0.1],
                               reco_shape, dtype=np.float32)

tilted = TiltedBookletsGeometry(apart, dpart, src_radius=1.2,
                                det_radius=1, pitch=0.05)


# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(reco_space, modified=True)

# ray transform
ray_trafo = odl.tomo.RayTransform(reco_space, tilted)

# Create projection data by calling the ray transform on the phantom
with timer('FW Ray Transform'):
    proj_data = ray_trafo(phantom).asarray()

with timer('Siemens BP - float32'):
    rec = siemens_bp(reco_space=reco_space, geometry=tilted,
                     proj_data=proj_data)

t_proj_data = torch.tensor(proj_data, dtype=torch.float16,
                           device=torch.device('cuda'))
torch.cuda.synchronize()
with timer('Siemens BP - float16, GPU'):
    rec = siemens_bp(reco_space=reco_space, geometry=tilted,
                     proj_data=t_proj_data, dtype=torch.float16, in_mem=True)


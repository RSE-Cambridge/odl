"""Example using the Siemens BP on 3d tilted booklets geometry.

After initialising a ``TiltedBookletsGeometry`` object, three
reconstruction methods are compared:
- Adjoint Ray transform;
- Siemens BP with default arguments;
- Siemens BP with half precision and I/O data on GPU.
Finally, the central slices of the original object and the three
reconstructions are displayed.
"""

import torch
import numpy as np
import odl

import matplotlib.pyplot as plt
from odl.util.testutils import timer
from odl.tomo.geometry import TiltedBookletsGeometry
from odl.tomo.backends import siemens_bp


# Define geometry parameters

n_angles = 500
reco_shape = (512,512,24)
det_shape = (736, 64)

print(f'Angles: {n_angles}')
print(f'Reconstruction space shape: {reco_shape}')
print(f'Detecor shape: {det_shape}')

# Define angle, detector and reconstruction space partitions
apart = odl.uniform_partition(0,4*np.pi, n_angles)
dpart = odl.uniform_partition([-1, -0.1], [1, .1], (736, 64)) 

reco_space = odl.uniform_discr([-0.25, -0.25, 0], [0.25, 0.25, 0.1],
                               reco_shape, dtype=np.float32)

# Define the geometry
tilted = TiltedBookletsGeometry(apart, dpart, src_radius=1.2,
                                det_radius=1, pitch=0.05)


# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(reco_space, modified=True)

# define ray transform operator
ray_trafo = odl.tomo.RayTransform(reco_space, tilted)

# Create projection data by calling the ray transform on the phantom
with timer('FW Ray Transform'):
    proj_data = ray_trafo(phantom).asarray()

# Perform reconstruction via adjoint ray transform
with timer('BW Adjoint Ray Transform'):
    rec_rt = ray_trafo.adjoint(proj_data).asarray()

# Call siemens_bp with minimum requirements
with timer('Siemens BP, float32'):
    rec = siemens_bp(reco_space=reco_space, geometry=tilted,
                     proj_data=proj_data)

# To improve perfomances, keep data on GPU and halve precision
# Move input data on GPU
cuda = torch.device('cuda')
t_proj_data = torch.tensor(proj_data, dtype=torch.float16, device=cuda)
torch.cuda.synchronize()

with timer('Siemens BP, float16, I/O GPU'):
    # halve precision, keep output data on GPU
    rec_gpu = siemens_bp(reco_space=reco_space, geometry=tilted,
                         proj_data=t_proj_data, dtype=torch.float16,
                         out_device=cuda)

# Display results for comparison
coords = (slice(None), slice(None), reco_shape[-1]//2)

plt.figure('Phantom')
plt.imshow(phantom.asarray()[coords], origin='lower', cmap='bone')
plt.figure('Adjoint Ray Transform')
plt.imshow(rec_rt[coords], origin='lower', cmap='bone')
plt.figure('Siemens BP Reconstruction')
plt.imshow(rec[coords], origin='lower', cmap='bone')
plt.figure('Siemens BP Reconstruction I/O GPU')
plt.imshow(rec_gpu.cpu().to(torch.float32)[coords], origin='lower', cmap='bone')
plt.show()
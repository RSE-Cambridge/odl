"""Example using the Siemens BP on 3d tilted booklets geometry."""

import torch
import numpy as np
import odl

import torch.autograd.profiler as profiler
from time import perf_counter
from odl.tomo.geometry.siemens import TiltedBookletsGeometry
# from odl.tomo.operators.w_ import *
from odl.tomo.operators.wfbp_torch import *

import sys
print(sys.version)
print(odl.__version__)

def benchmark_f(f, name, dtype, recon_space, geometry, proj_data, **kwargs):
    print(f'{name.ljust(20)[:20]}', end='\t')
    print(f'{dtype}', end='\t')
    print(f'{"default" if not "t_chunk" in kwargs else kwargs.get("t_chunk")}', end='\t')

    start = perf_counter()
            
    _ = f(recon_space, geometry=geometry, proj_data=proj_data, dtype=dtype, **kwargs)
    torch.cuda.synchronize()

    stop = perf_counter()
    print(f'{(stop-start):.2f} s', end='\t')

    print(f'{torch.cuda.max_memory_allocated(0)//(2**20)}'.rjust(6)+
        f' / {torch.cuda.memory_reserved(0)//(2**20)}  MB', end='\t')
    print(f'Allocated: {torch.cuda.memory_allocated(0)//(2**20)} MB', end='\n')
    torch.cuda.reset_peak_memory_stats(0)


n_angles = 8000
recon_shape = (512,512,24)
proj_filename = f"data/{n_angles}_{'_'.join( [str(int) for int in recon_shape])}.npy"

print(f'Angles: {n_angles}')
print(f'Reconstruction space: {recon_shape}')
print(f'Projection data filename: {proj_filename}')

apart = odl.uniform_partition(0,4*np.pi, n_angles)
dpart = odl.uniform_partition([-1, -0.1], [1, .1], (736, 64)) 

reco_space = odl.uniform_discr([-0.25, -0.25, 0], [0.25, 0.25, 0.1],
                                    recon_shape, dtype = np.float32)

tilted = TiltedBookletsGeometry(apart, dpart, src_radius=1.2, det_radius=1, pitch=0.05)


# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(reco_space, modified=True)

# ray transform
ray_trafo = odl.tomo.RayTransform(reco_space, tilted)

try:
    proj_data = np.load(proj_filename)
    print('Propjection data loaded.')
except:
    print('Propjection data not found. Computing it...')
    # Create projection data by calling the ray transform on the phantom
    start = perf_counter()
    proj_data = ray_trafo(phantom).asarray()
    stop = perf_counter()
    print(f'FW done after:\t\t{(stop-start):5.3f} seconds')
    np.save(proj_filename, proj_data)


#### TORCH
wfbp_torch_z_zxy_tuv_full
test_list = [(wfbp_torch_z, 'wfbp_torch_z', torch.float16, [10, 50, 100, 500, 1000]),
            (wfbp_torch_z, 'wfbp_torch_z', torch.float32, [10, 50, 100, 500]),
            (wfbp_torch_angles, 'wfbp_torch_angles', torch.float16, [10, 50, 100, 500, 1000]),
            (wfbp_torch_angles, 'wfbp_torch_angles', torch.float32, [10, 50, 100, 500]),
            (wfbp_torch_full, 'wfbp_torch_full', torch.float16, [10, 20, 30, 40]),
            (wfbp_torch_full, 'wfbp_torch_full', torch.float32, [10, 20, 30, 40]),
            (wfbp_torch_z_zxy, 'wfbp_torch_z_zxy', torch.float16, [10, 50, 100, 500, 1000]),
            (wfbp_torch_z_zxy, 'wfbp_torch_z_zxy', torch.float32, [10, 50, 100, 500]),
            (wfbp_torch_z_zxy_tuv, 'wfbp_torch_z_zxy_tuv', torch.float16, [10, 50, 100, 500]),
            (wfbp_torch_z_zxy_tuv, 'wfbp_torch_z_zxy_tuv', torch.float32, [10, 50, 100, 200]),
            (wfbp_torch_z_zxy_tuv_full, 'wfbp_torch_z_zxy_tuv_full', torch.float16, [10, 20, 30, 40]),
            (wfbp_torch_z_zxy_tuv_full, 'wfbp_torch_z_zxy_tuv_full', torch.float32, [10, 20, 30, 40])]

test_list = [(wfbp_torch_z_zxy_tuv, 'wfbp_torch_z_zxy_tuv', torch.float16, [10, 50, 100, 500, 1000]),
            (wfbp_torch_z_zxy_tuv, 'wfbp_torch_z_zxy_tuv', torch.float32, [10, 50, 100, 200]),
            (wfbp_torch_z_zxy, 'wfbp_torch_z_zxy', torch.float16, [10, 50, 100, 500, 1000]),
            (wfbp_torch_z_zxy, 'wfbp_torch_z_zxy', torch.float32, [10, 50, 100, 500])]

_proj_data = proj_data
print(f'\nPROJ SHAPE: {_proj_data.shape}')

print(f'\nLOAD TO GPU')
for f, name, dtype, t_chunks in test_list:
    for t_chunk in t_chunks:
        benchmark_f(f, name, dtype, reco_space,
                    geometry=tilted, proj_data=_proj_data, t_chunk=t_chunk)

print(f'\nIN MEMORY')
for f, name, dtype, t_chunks in test_list:
    for t_chunk in t_chunks:
        t_proj_data = torch.tensor(_proj_data, dtype=dtype, device=torch.device('cuda'))
        torch.cuda.synchronize()
        benchmark_f(f, name, dtype, reco_space, geometry=tilted,
                    proj_data=t_proj_data, in_mem=True, t_chunk=t_chunk)
            
        del t_proj_data


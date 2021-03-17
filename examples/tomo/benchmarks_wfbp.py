"""Example using the Siemens BP on 3d tilted booklets geometry."""

import torch
import numpy as np
import odl

import torch.autograd.profiler as profiler
from time import perf_counter
from odl.tomo.geometry.siemens import TiltedBookletsGeometry
from odl.tomo.operators.w_ import *
from odl.tomo.operators.wfbp_torch import *

import sys
print(sys.version)
print(odl.__version__)

def benchmark_f(f, name, dtype, recon_space, geometry, proj_data, **kwargs):
    print(f'{name.ljust(20)[:20]}', end='\t')
    print(f'{dtype}', end='\t')

    start = perf_counter()
            
    _ = f(recon_space, geometry=geometry, proj_data=proj_data, dtype=dtype, **kwargs)
    torch.cuda.synchronize()

    stop = perf_counter()
    print(f'{(stop-start):.3f} s', end='\t')

    print(f'{torch.cuda.max_memory_allocated(0)//(2**20)}'.rjust(6)+
        f' / {torch.cuda.memory_reserved(0)//(2**20)}  MB', end='\t')
    print(f'Allocated: {torch.cuda.memory_allocated(0)//(2**20)} MB', end='\n')
    torch.cuda.reset_peak_memory_stats(0)


n_angles = 1000
recon_shape = (512,512,24)

print('Angles:', n_angles)
print('Reconstruction space:', recon_shape)


apart = odl.uniform_partition(0,4*np.pi, n_angles)
dpart = odl.uniform_partition([-1, -0.1], [1, .1], (736, 64)) 

reco_space = odl.uniform_discr([-0.25, -0.25, 0], [0.25, 0.25, 0.1],
                                    recon_shape, dtype = np.float32)

tilted = TiltedBookletsGeometry(apart, dpart, src_radius=1.2, det_radius=1, pitch=0.05)


# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(reco_space, modified=True)

# ray transform
ray_trafo = odl.tomo.RayTransform(reco_space, tilted)

# Create projection data by calling the ray transform on the phantom
start = perf_counter()
proj_data = ray_trafo(phantom).asarray()
stop = perf_counter()
print(f'FW done after:\t\t{(stop-start):.3f} seconds')


#### TORCH

test_list = [(wfbp_angles_proj_chunk, 'ANGLES Proj chunk', torch.float16),
            (wfbp_angles_proj_chunk, 'ANGLES Proj chunk', torch.float32),
            (wfbp_full_torch, 'ORIGINAL', torch.float16),
            (wfbp_full_torch, 'ORIGINAL', torch.float32),
            (wfbp_angles, 'ANGLES', torch.float16),
            (wfbp_angles, 'ANGLES', torch.float32),
            (w_a_last, 'Angles LAST', torch.float16),
            (w_a_last, 'Angles LAST', torch.float32),
            (w_a_first, 'Angles FIRST', torch.float16),
            (w_a_first, 'Angles FIRST', torch.float32),
            (w_angles, 'LOOP ON ANGLES', torch.float16),
            (w_angles, 'LOOP ON ANGLES', torch.float32)]


_proj_data = proj_data
print(f'\nPROJ SHAPE: {_proj_data.shape}')

print(f'\nLOAD TO GPU')
for f, name, dtype in test_list:
    benchmark_f(f, name, dtype, reco_space, geometry=tilted, proj_data=_proj_data)

print(f'\nIN MEMORY')

for f, name, dtype in test_list:
    t_proj_data = torch.tensor(_proj_data, dtype=dtype, device=torch.device('cuda'))
    torch.cuda.synchronize()
    benchmark_f(f, name, dtype, reco_space, geometry=tilted,
                proj_data=t_proj_data, in_mem=True)
    del t_proj_data

### (u, theta, v)
_proj_data = np.moveaxis(proj_data, 0, 1).copy()
print(f'\nPROJ SHAPE: {_proj_data.shape}')

test_list = [(w_utv, 'LAST (u, theta, v)', torch.float16),
            (w_utv, 'LAST (u, theta, v)', torch.float32)]

# test_list = []

print(f'\nLOAD TO GPU')
for f, name, dtype in test_list:
    benchmark_f(f, name, dtype, reco_space, geometry=tilted, proj_data=_proj_data)

print(f'\nIN MEMORY')

for f, name, dtype in test_list:
    t_proj_data = torch.tensor(_proj_data, dtype=dtype, device=torch.device('cuda'))
    torch.cuda.synchronize()
    benchmark_f(f, name, dtype, reco_space, geometry=tilted,
                proj_data=t_proj_data, in_mem=True)
    del t_proj_data


### (v, theta, u)
_proj_data = np.moveaxis(proj_data, 2, 0).copy()
print(f'\nPROJ SHAPE: {_proj_data.shape}')

test_list = [(w_vtu, 'LAST (v, theta, u)', torch.float16),
            (w_vtu, 'LAST (v, theta, u)', torch.float32),
            (w_angles_vtu, 'ANGLES (v, theta, u)', torch.float16),
            (w_angles_vtu, 'ANGLES (v, theta, u)', torch.float32)]

# test_list = []
print(f'\nLOAD TO GPU')
for f, name, dtype in test_list:
    benchmark_f(f, name, dtype, reco_space, geometry=tilted, proj_data=_proj_data)

print(f'\nIN MEMORY')

for f, name, dtype in test_list:
    t_proj_data = torch.tensor(_proj_data, dtype=dtype, device=torch.device('cuda'))
    torch.cuda.synchronize()
    benchmark_f(f, name, dtype, reco_space, geometry=tilted,
                proj_data=t_proj_data, in_mem=True)
    del t_proj_data


### (theta, v, u)
_proj_data = np.moveaxis(proj_data, 2, 1).copy()
print(f'\nPROJ SHAPE: {_proj_data.shape}')

test_list = [(w_angles_tvu, 'ANGLES (theta, v, u)', torch.float16),
            (w_angles_tvu, 'ANGLES (theta, v, u)', torch.float32)]

# test_list = []
print(f'\nLOAD TO GPU')
for f, name, dtype in test_list:
    benchmark_f(f, name, dtype, reco_space, geometry=tilted, proj_data=_proj_data)

print(f'\nIN MEMORY')

for f, name, dtype in test_list:
    t_proj_data = torch.tensor(_proj_data, dtype=dtype, device=torch.device('cuda'))
    torch.cuda.synchronize()
    benchmark_f(f, name, dtype, reco_space, geometry=tilted,
                proj_data=t_proj_data, in_mem=True)
    del t_proj_data







##### PROFILER

# with profiler.profile(with_stack=True, profile_memory=True, use_cuda=True) as prof:
#     with profiler.record_function("angles last"):
#         w_a_last(reco_space, geometry=tilted, proj_data=proj_data)

# with open('w_a_last.prof', 'w') as out:
#     out.write(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_time_total"))  # , row_limit=10

# prof.export_chrome_trace("w_a_last.json")

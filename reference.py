import sgl
import slangpy as spy
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import utils

probe_data = utils.load_exr("probe.exr")
img_w = probe_data.shape[0]
img_h = probe_data.shape[1]
alpha_channel = np.ones((img_w, img_h, 1), dtype=probe_data.dtype)
probe_data = np.concatenate([probe_data, alpha_channel], axis=2)
probe_data = probe_data.reshape(-1, 1, 1)

device = spy.create_device(include_paths=[
    pathlib.Path(__file__).parent.absolute(),
])

radiance_map = device.create_texture(width= img_w, height = img_h, format = sgl.Format.rgba32_float,
                                  usage= sgl.ResourceUsage.shader_resource | sgl.ResourceUsage.unordered_access,
                                  data = probe_data)

module_ibl = spy.Module.load_from_file(device, "IBL.slang")

mip_count = 5
mip_level = 1
linear_roughness = mip_level / float(mip_count - 1)
alpha = linear_roughness * linear_roughness
prefiltered_map_w = img_w >> mip_level
prefiltered_map_h = img_h >> mip_level

dispatch_dimesnion = np.zeros((prefiltered_map_w, prefiltered_map_h, 4), dtype=np.float32)
thread_index = spy.grid(shape=(prefiltered_map_w, prefiltered_map_h))

prefiltered_map_w_tensor = np.full((prefiltered_map_w, prefiltered_map_h), prefiltered_map_w, dtype=np.uint32)
prefiltered_map_h_tensor = np.full((prefiltered_map_w, prefiltered_map_h), prefiltered_map_h, dtype=np.uint32)
alpha_tensor = np.full((prefiltered_map_w, prefiltered_map_h), alpha, dtype=np.float32)
mip_level_tensor = np.full((prefiltered_map_w, prefiltered_map_h), mip_level, dtype=np.int32)

linear_sampler = device.create_sampler()

prefiltered_map = module_ibl.PrefilterEnvmap(thread_index, 
                                             alpha_tensor, mip_level_tensor, 
                                             radiance_map, 
                                             _result=dispatch_dimesnion)
plt.imshow(prefiltered_map)
plt.show()


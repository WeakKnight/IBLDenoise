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
mip_level = 3
sample_count = 64
sample_count_reference = 1024

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
sample_count_tensor = np.full((prefiltered_map_w, prefiltered_map_h), sample_count, dtype=np.int32)
sample_count_reference_tensor = np.full((prefiltered_map_w, prefiltered_map_h), sample_count_reference, dtype=np.int32)

linear_sampler = device.create_sampler()

prefiltered_map_data = module_ibl.PrefilterEnvmap(thread_index, 
                                             alpha_tensor, mip_level_tensor,
                                             sample_count_tensor, 
                                             radiance_map, 
                                             _result=dispatch_dimesnion)
prefiltered_map_data_copy = prefiltered_map_data.copy()
prefiltered_map = device.create_texture(width= prefiltered_map_w, height = prefiltered_map_h, format = sgl.Format.rgba32_float,
                                  usage= sgl.ResourceUsage.shader_resource | sgl.ResourceUsage.unordered_access,
                                  data = prefiltered_map_data)

prefiltered_map_reference_data = module_ibl.PrefilterEnvmap(thread_index, 
                                             alpha_tensor, mip_level_tensor,
                                             sample_count_reference_tensor, 
                                             radiance_map, 
                                             _result=dispatch_dimesnion)
prefiltered_map_reference_data_copy = prefiltered_map_reference_data.copy()

blurred_prefiltered_map_data = module_ibl.BoxBlur(thread_index, 
                        prefiltered_map,
                        prefiltered_map_w_tensor,
                        prefiltered_map_h_tensor,
                        _result=dispatch_dimesnion)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# 在每个子图中显示图像
axes[0].imshow(prefiltered_map_data_copy)
axes[0].set_title("Raw")
axes[0].axis("off")  # 关闭坐标轴显示

axes[1].imshow(blurred_prefiltered_map_data)
axes[1].set_title("Blurred")
axes[1].axis("off")

axes[2].imshow(prefiltered_map_reference_data_copy)
axes[2].set_title("Reference")
axes[2].axis("off")

plt.tight_layout()
plt.show()

# plt.imshow(prefiltered_map_data_copy)
# plt.show()


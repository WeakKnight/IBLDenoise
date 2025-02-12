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

probe_tex = device.create_texture(width= img_w, height = img_h, format = sgl.Format.rgba32_float,
                                  usage= sgl.ResourceUsage.shader_resource | sgl.ResourceUsage.unordered_access,
                                  data = probe_data)

output_tex = device.create_texture(width= img_w, height = img_h, format = sgl.Format.rgba8_unorm,
                                  usage= sgl.ResourceUsage.shader_resource | sgl.ResourceUsage.unordered_access)

module = spy.Module.load_from_file(device, "reference.slang")
module.aces_filmic(probe_tex, output_tex)

output_bitmap = output_tex.to_numpy()
plt.imshow(output_bitmap)
plt.show()


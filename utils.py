import OpenEXR
import Imath
import numpy as np

def load_exr(file_path):
    # 打开 EXR 文件
    exr_file = OpenEXR.InputFile(file_path)

    # 获取数据窗口，确定图像尺寸
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    print("header", header)

    # 定义像素数据类型，这里使用 FLOAT（32位浮点数）
    pt = Imath.PixelType(Imath.PixelType.HALF)

    # 读取各通道数据
    # 示例中假设 EXR 文件含有 "R", "G", "B" 三个通道，如果需要 Alpha 则加入 "A"
    channels = ["R", "G", "B"]
    data = {}
    for c in channels:
        # 读取通道数据，返回的是字节流
        ch_str = exr_file.channel(c, pt)
        # 将字节数据转换为 numpy 数组，并指定数据类型为 float32
        data[c] = np.frombuffer(ch_str, dtype=np.float16)
        # 调整形状为 (height, width)
        data[c] = np.reshape(data[c], (height, width))

    # 将各通道数组沿最后一个轴堆叠，得到形状为 (height, width, 3) 的数组
    img = np.stack([data[c] for c in channels], axis=-1)
    img = img.astype(np.float32)
    return img
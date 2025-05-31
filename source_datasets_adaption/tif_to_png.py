import os
from PIL import Image
import tqdm

import matplotlib.pyplot as plt
import rasterio
import numpy as np

img_dir = "images"
for filename in tqdm.tqdm(os.listdir(img_dir)):
    if filename.lower().endswith(".tif"):
        tif_path = os.path.join(img_dir, filename)
        png_path = os.path.splitext(tif_path)[0] + ".png"
        try:
            img = Image.open(tif_path)
            img.save(png_path)
        except Exception as e:
            print(f"普通方式打开失败：{tif_path}，尝试rasterio，错误信息：{e}")
            try:
                with rasterio.open(tif_path) as src:
                    img = src.read([1, 2, 3])  # 取前三个波段
                    img = np.moveaxis(img, 0, -1)
                # 对每个波段分别做2%~98%分位拉伸
                img_stretch = np.zeros_like(img, dtype=np.float32)
                for i in range(3):
                    p2 = np.percentile(img[..., i], 2)
                    p98 = np.percentile(img[..., i], 98)
                    img_stretch[..., i] = np.clip(
                        (img[..., i] - p2) / (p98 - p2 + 1e-8), 0, 1
                    )
                # 保存为png
                plt.imsave(png_path, img_stretch)
            except Exception as e2:
                raise RuntimeError(
                    f"rasterio方式也失败：{tif_path}，错误信息：{e2}"
                ) from e2


def replace_tif_with_png(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    new_lines = [line.replace(".tif", ".png") for line in lines]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


# 修改 train.txt 和 val.txt
replace_tif_with_png("train.txt")
replace_tif_with_png("val.txt")

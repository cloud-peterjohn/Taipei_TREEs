import os
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from PIL import Image
import numpy as np
from tqdm import tqdm

base_dir = r"d:\Applications\Downloads\IDTREES_competition_train_v2\train"
itc_dir = os.path.join(base_dir, "ITC")
field_dir = os.path.join(base_dir, "Field")
remote_dir = os.path.join(base_dir, r"RemoteSensing\RGB")
save_img_dir = os.path.join(base_dir, "tree_crops")
os.makedirs(save_img_dir, exist_ok=True)

df_taxon = pd.read_csv(os.path.join(field_dir, "taxonID_ScientificName.csv"))
taxonid2classid = {
    row["taxonID"]: int(row["taxonCode"]) - 1 for _, row in df_taxon.iterrows()
}
taxonid2name = {row["taxonID"]: row["scientificName"] for _, row in df_taxon.iterrows()}

df_train = pd.read_csv(
    os.path.join(field_dir, "train_data.csv"), dtype={"indvdID": str, "taxonID": str}
)
df_rs = pd.read_csv(os.path.join(field_dir, "itc_rsFile.csv"))

shp_files = [f for f in os.listdir(itc_dir) if f.endswith(".shp")]
gdf_list = []
for shp_file in shp_files:
    gdf = gpd.read_file(os.path.join(itc_dir, shp_file))
    gdf_list.append(gdf)
gdf = pd.concat(gdf_list, ignore_index=True)

gdf = gdf.merge(df_rs, on="indvdID", how="left")
gdf = gdf.merge(df_train[["indvdID", "taxonID"]], on="indvdID", how="left")
gdf = gdf[~gdf["rsFile"].isna()]
gdf = gdf[~gdf["taxonID"].isna()]

gdf = gdf[gdf["rsFile"].str.startswith(("MLBS", "OSBS"))]

result = []
img_idx = 0

for rsfile, group in tqdm(gdf.groupby("rsFile"), desc="Processing images"):
    tif_path = os.path.join(remote_dir, rsfile)
    if not os.path.exists(tif_path):
        continue
    with rasterio.open(tif_path) as src:
        if group.crs != src.crs:
            group = group.to_crs(src.crs)
        width = src.width
        height = src.height
        transform = src.transform
        for _, row in group.iterrows():
            minx, miny, maxx, maxy = row.geometry.bounds
            x0, y0 = ~transform * (minx, miny)
            x1, y1 = ~transform * (maxx, maxy)
            x0, x1 = sorted([int(np.floor(x0)), int(np.ceil(x1))])
            y0, y1 = sorted([int(np.floor(y0)), int(np.ceil(y1))])
            x0 = max(0, min(width - 1, x0))
            x1 = max(0, min(width, x1))
            y0 = max(0, min(height - 1, y0))
            y1 = max(0, min(height, y1))
            if x1 <= x0 or y1 <= y0:
                continue
            window = Window(x0, y0, x1 - x0, y1 - y0)
            img = src.read(window=window)
            img = np.transpose(img, (1, 2, 0))
            if img.dtype != np.uint8:
                img = np.clip(img / img.max() * 255, 0, 255).astype(np.uint8)
            img_name = f"{img_idx}.png"
            img_save_path = os.path.join(save_img_dir, img_name)
            Image.fromarray(img).save(img_save_path)
            taxonid = row["taxonID"]
            classid = taxonid2classid.get(taxonid, -1)
            classname = taxonid2name.get(taxonid, "Unknown")
            result.append([img_name, classid, classname])
            img_idx += 1

result_df = pd.DataFrame(result, columns=["filename", "class_id", "scientificName"])
result_df.to_csv(
    os.path.join(base_dir, "tree_crops_labels.csv"), index=False, encoding="utf-8"
)
print(f"Saved {img_idx} images, label file: tree_crops_labels.csv")

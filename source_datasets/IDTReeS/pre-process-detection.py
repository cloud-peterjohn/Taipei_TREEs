# Original datasets are downloaded from https://zenodo.org/records/3934932
# Graves, S., & Marconi, S. (2020). IDTReeS 2020 Competition Data (版 4) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3934932


"""
Generate YOLO dataset with classfication labels.
"""

# import geopandas as gpd
# import pandas as pd
# import os
# import rasterio
# import shutil

# # Paths
# itc_dir = r"d:\Applications\Downloads\IDTREES_competition_train_v2\train\ITC"
# field_dir = r"d:\Applications\Downloads\IDTREES_competition_train_v2\train\Field"
# remote_dir = (
#     r"d:\Applications\Downloads\IDTREES_competition_train_v2\train\RemoteSensing\RGB"
# )
# save_dir = r"d:\Applications\Downloads\IDTREES_competition_train_v2\train\yolo_dataset"
# labels_dir = os.path.join(save_dir, "labels")
# images_dir = os.path.join(save_dir, "images")
# os.makedirs(labels_dir, exist_ok=True)
# os.makedirs(images_dir, exist_ok=True)

# # Read all ITC shapefiles
# shp_files = [f for f in os.listdir(itc_dir) if f.endswith(".shp")]
# gdf_list = []
# for shp_file in shp_files:
#     gdf = gpd.read_file(os.path.join(itc_dir, shp_file))
#     gdf_list.append(gdf)
# gdf = pd.concat(gdf_list, ignore_index=True)

# # Read itc_rsFile.csv
# itc_rsfile = os.path.join(field_dir, "itc_rsFile.csv")
# df_rs = pd.read_csv(itc_rsfile)

# # Read train_data.csv for taxonID
# train_data_file = os.path.join(field_dir, "train_data.csv")
# df_train = pd.read_csv(train_data_file, dtype={"indvdID": str, "taxonID": str})

# # Read taxonID_ScientificName.csv for taxonID to class_id mapping
# taxonid_map_file = os.path.join(field_dir, "taxonID_ScientificName.csv")
# df_taxon = pd.read_csv(taxonid_map_file)
# taxonid2classid = {
#     row["taxonID"]: int(row["taxonCode"]) - 1 for _, row in df_taxon.iterrows()
# }  # YOLO class_id starts from 0

# # Merge to get image file and taxonID for each crown
# gdf = gdf.merge(df_rs, on="indvdID", how="left")
# gdf = gdf.merge(df_train[["indvdID", "taxonID"]], on="indvdID", how="left")
# gdf = gdf[~gdf["rsFile"].isna()]
# gdf = gdf[~gdf["taxonID"].isna()]


# def geo_to_pixel(x, y, transform):
#     col, row = ~transform * (x, y)
#     return col, row


# # Process each image
# for rsfile, group in gdf.groupby("rsFile"):
#     tif_path = os.path.join(remote_dir, rsfile)
#     if not os.path.exists(tif_path):
#         continue
#     with rasterio.open(tif_path) as src:
#         transform = src.transform
#         width = src.width
#         height = src.height

#         yolo_lines = []
#         for _, row in group.iterrows():
#             minx, miny, maxx, maxy = row.geometry.bounds
#             x_min, y_max = geo_to_pixel(minx, maxy, transform)
#             x_max, y_min = geo_to_pixel(maxx, miny, transform)
#             x_min = max(0, min(width - 1, x_min))
#             x_max = max(0, min(width - 1, x_max))
#             y_min = max(0, min(height - 1, y_min))
#             y_max = max(0, min(height - 1, y_max))
#             x_center = (x_min + x_max) / 2 / width
#             y_center = (y_min + y_max) / 2 / height
#             w = abs(x_max - x_min) / width
#             h = abs(y_max - y_min) / height
#             # Get class_id from taxonID
#             class_id = taxonid2classid.get(row["taxonID"], -1)
#             if class_id == -1:
#                 continue
#             yolo_lines.append(
#                 f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
#             )

#         # Save label file
#         txt_name = os.path.splitext(rsfile)[0] + ".txt"
#         txt_path = os.path.join(labels_dir, txt_name)
#         with open(txt_path, "w") as f:
#             f.write("\n".join(yolo_lines))

#         # Copy image file
#         img_dst = os.path.join(images_dir, rsfile)
#         if not os.path.exists(img_dst):
#             shutil.copy(tif_path, img_dst)

# print(
#     "YOLO dataset generation finished. Labels in:", labels_dir, "Images in:", images_dir
# )


"""
Generate YOLO dataset without classification labels.
"""
import geopandas as gpd
import pandas as pd
import os
import rasterio
import shutil

# Paths
itc_dir = r"d:\Applications\Downloads\IDTREES_competition_train_v2\train\ITC"
field_dir = r"d:\Applications\Downloads\IDTREES_competition_train_v2\train\Field"
remote_dir = (
    r"d:\Applications\Downloads\IDTREES_competition_train_v2\train\RemoteSensing\RGB"
)
save_dir = r"d:\Applications\Downloads\IDTREES_competition_train_v2\train\yolo_dataset"
labels_dir = os.path.join(save_dir, "labels")
images_dir = os.path.join(save_dir, "images")
os.makedirs(labels_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

# Read all ITC shapefiles
shp_files = [f for f in os.listdir(itc_dir) if f.endswith(".shp")]
gdf_list = []
for shp_file in shp_files:
    gdf = gpd.read_file(os.path.join(itc_dir, shp_file))
    gdf_list.append(gdf)
gdf = pd.concat(gdf_list, ignore_index=True)

# Read itc_rsFile.csv
itc_rsfile = os.path.join(field_dir, "itc_rsFile.csv")
df_rs = pd.read_csv(itc_rsfile)

# Read train_data.csv for taxonID
train_data_file = os.path.join(field_dir, "train_data.csv")
df_train = pd.read_csv(train_data_file, dtype={"indvdID": str, "taxonID": str})

# Read taxonID_ScientificName.csv for taxonID to class_id mapping
taxonid_map_file = os.path.join(field_dir, "taxonID_ScientificName.csv")
df_taxon = pd.read_csv(taxonid_map_file)
# 所有tree都归为一个类，class_id恒为0
taxonid2classid = {row["taxonID"]: 0 for _, row in df_taxon.iterrows()}

# Merge to get image file and taxonID for each crown
gdf = gdf.merge(df_rs, on="indvdID", how="left")
gdf = gdf.merge(df_train[["indvdID", "taxonID"]], on="indvdID", how="left")
gdf = gdf[~gdf["rsFile"].isna()]
gdf = gdf[~gdf["taxonID"].isna()]


def geo_to_pixel(x, y, transform):
    col, row = ~transform * (x, y)
    return col, row


# Process each image
for rsfile, group in gdf.groupby("rsFile"):
    tif_path = os.path.join(remote_dir, rsfile)
    if not os.path.exists(tif_path):
        continue
    with rasterio.open(tif_path) as src:
        transform = src.transform
        width = src.width
        height = src.height

        yolo_lines = []
        for _, row in group.iterrows():
            minx, miny, maxx, maxy = row.geometry.bounds
            x_min, y_max = geo_to_pixel(minx, maxy, transform)
            x_max, y_min = geo_to_pixel(maxx, miny, transform)
            x_min = max(0, min(width - 1, x_min))
            x_max = max(0, min(width - 1, x_max))
            y_min = max(0, min(height - 1, y_min))
            y_max = max(0, min(height - 1, y_max))
            x_center = (x_min + x_max) / 2 / width
            y_center = (y_min + y_max) / 2 / height
            w = abs(x_max - x_min) / width
            h = abs(y_max - y_min) / height
            # Get class_id from taxonID
            class_id = taxonid2classid.get(row["taxonID"], 0)  # 只有一个类，恒为0
            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            )

        # Save label file
        txt_name = os.path.splitext(rsfile)[0] + ".txt"
        txt_path = os.path.join(labels_dir, txt_name)
        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_lines))

        # Copy image file
        img_dst = os.path.join(images_dir, rsfile)
        if not os.path.exists(img_dst):
            shutil.copy(tif_path, img_dst)

print(
    "YOLO dataset generation finished. Labels in:", labels_dir, "Images in:", images_dir
)

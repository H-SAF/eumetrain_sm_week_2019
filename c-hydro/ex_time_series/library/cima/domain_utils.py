import json
import rasterio

from os.path import join
from rasterio import features
from rasterio.transform import from_origin
import rasterio.mask
import geopandas as gpd
import pandas as pd
import numpy as np
import tempfile

from shapely.geometry import Point

import rasterio
import numpy as np
from pygeogrids.grids import BasicGrid

def get_grid(file_name):

    data_handle = rasterio.open(file_name)
    data_bbox = data_handle.bounds
    data_res = data_handle.res

    lon_1d = np.arange(data_bbox.left, data_bbox.right + np.abs(data_res[0] / 2), np.abs(data_res[0]), float)
    lat_1d = np.arange(data_bbox.bottom, data_bbox.top + np.abs(data_res[1] / 2), np.abs(data_res[1]), float)
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

    grid = BasicGrid(lon_2d.flatten(), lat_2d.flatten()).to_cell_grid(cellsize=5.)

    return grid, lon_2d, lat_2d, data_bbox

def create_points_shp(data, x, y, data_format="int64",
                      epsg="EPSG:4326", file_name_shp="points.shp"):

    points_rows = np.array([x, y, data])
    points_cols = np.transpose(points_rows)

    assert np.all(points_rows[0, :] == points_cols[:, 0])
    assert np.all(points_rows[1, :] == points_cols[:, 1])
    assert np.all(points_rows[2, :] == points_cols[:, 2])

    points_df = pd.DataFrame(points_cols, columns=["x", "y", "z"])
    points_df.head()
    points_df = points_df.astype(dtype={"x": "float64", "y": "float64", "z": data_format})

    points_df['geometry'] = points_df.apply(lambda row: Point(row.x, row.y, row.z), axis=1)
    points_df = points_df.drop(["x", "y", "z"], axis=1)
    # create the GeoDatFrame
    points_crs = {'init': epsg}
    points_gdf = gpd.GeoDataFrame(points_df, crs=points_crs, geometry=points_df.geometry)
    # save the GeoDataFrame
    points_gdf.to_file(driver='ESRI Shapefile', filename=file_name_shp)

# Method to create mask
def create_mask(rows, cols, geoms, transform, epsg="EPSG:4326", file_name_masked="masked.tiff"):

    mask = rasterio.features.rasterize(geoms,
                                       out_shape=[rows, cols],
                                       transform=transform,
                                       dtype=rasterio.int32
                                       )

    dst_mask = rasterio.open(file_name_masked, 'w', driver='GTiff',
                                 height=rows, width=cols,
                                 count=1, dtype=rasterio.int32,
                                 crs=epsg,
                                 transform=transform)
    dst_mask.write(mask, 1)
    dst_mask.close()

# Method to create template file
def create_template_file(rows, cols, transform, epsg="EPSG:4326",
                         file_name_template="template.tiff"):

    arr = np.zeros([rows, cols], dtype=rasterio.int32)
    arr[:, :] = 1
    dst_template = rasterio.open(file_name_template, 'w', driver='GTiff',
                                 height=rows, width=cols,
                                 count=1, dtype=rasterio.int32,
                                 crs=epsg,
                                 transform=transform)
    dst_template.write(arr, 1)
    dst_template.close()

# Method to get file shp
def get_file_shp(file_name_shp, file_name_mask="masked.tiff", folder_tmp=tempfile.mkdtemp(),
            cell_size=0.0005, epsg="EPSG:4326", bbox_ext=0):

    shp = gpd.read_file(file_name_shp)
    geoms = ((feature['geometry'], 1) for feature in shp.iterfeatures())

    if 'AREA' in shp:
        area = shp['AREA'][0]
    else:
        area = None

    if bbox_ext > 1:
        bbox_ext = 1

    bbox_minx_shp = shp.bounds['minx'][0] - bbox_ext
    bbox_maxx_shp = shp.bounds['maxx'][0] + bbox_ext
    bbox_miny_shp = shp.bounds['miny'][0] - bbox_ext
    bbox_maxy_shp = shp.bounds['maxy'][0] + bbox_ext

    # Create geographical information
    geox_1d = np.arange(bbox_minx_shp, bbox_maxx_shp + np.abs(cell_size / 2), np.abs(cell_size), float)
    geoy_1d = np.arange(bbox_miny_shp, bbox_maxy_shp + np.abs(cell_size / 2), np.abs(cell_size), float)

    bbox_minx = np.min(geox_1d)
    bbox_maxx = np.max(geox_1d)
    bbox_miny = np.min(geoy_1d)
    bbox_maxy = np.max(geoy_1d)

    geox_2d, geoy_2d = np.meshgrid(geox_1d, geoy_1d)
    rows = int(np.round((bbox_maxy - bbox_miny) / cell_size + 1))
    cols = int(np.round((bbox_maxx - bbox_minx) / cell_size + 1))

    corner_ll_x = bbox_minx - cell_size / 2.0
    corner_ll_y = bbox_maxy + cell_size / 2.0

    # Create template raster
    transform = from_origin(corner_ll_x, corner_ll_y, cell_size, cell_size)
    create_template_file(rows, cols, transform, epsg,
                         file_name_template=join(folder_tmp, "template.tiff"))

    # Load template raster
    dst_reference = rasterio.open(join(folder_tmp, "template.tiff"))
    meta_reference = dst_reference.meta.copy()
    meta_reference.update(compress='lzw')
    dst_reference.close()

    # Create mask
    create_mask(rows, cols, geoms, meta_reference['transform'], epsg,
                file_name_masked=file_name_mask)

    return rows, cols, epsg, transform, meta_reference

# Method to get file json
def get_file_json(file_name_json):
    with open(file_name_json, 'r') as file_handle:
        file_data = json.load(file_handle)
    return file_data

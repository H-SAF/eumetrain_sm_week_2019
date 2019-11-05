import rasterio
import numpy as np
from scipy.interpolate import griddata

from pygeogrids.grids import BasicGrid, genreg_grid

# Method to create image, lons and lats
def create_image(lon, lat, data,
                 minlat=-90., maxlat=90., minlon=-180., maxlon=180.):

    other = BasicGrid(lon, lat)
    reg_grid = genreg_grid(0.1, 0.1, minlat=minlat, maxlon=maxlon,
                           maxlat=maxlat, minlon=minlon)
    lons = np.arange(minlon, maxlon, 0.1)
    lats = np.arange(minlat, maxlat, 0.1)

    lut = reg_grid.calc_lut(other, max_dist=25000)
    img = np.ma.masked_where(lut == -1, data[lut])

    img = np.flipud(img.reshape(reg_grid.shape[0], -1))

    return img, lons, lats

# Method to create map in tiff format
def create_map(value, rows, cols, epsg, transform, file_name_data='data.tiff', file_name_mask=None):

    if file_name_mask:
        dst_mask = rasterio.open(file_name_mask)
        mask = dst_mask.read(1)
        dst_mask.close()

        value[np.where(mask == 0)] = np.nan

    value_unique = np.unique(value)

    dst_value = rasterio.open(file_name_data, 'w', driver='GTiff',
                                 height=rows, width=cols,
                                 count=1, dtype=rasterio.float64,
                                 crs=epsg,
                                 transform=transform)
    dst_value.write(value, 1)
    dst_value.close()

# Method to interpolate points to grid
def interpolate_point2map(lons_in, lats_in, values_in, lons_out, lats_out, nodata=-9999, interp='nearest'):

    values_out = griddata((lons_in.ravel(), lats_in.ravel()),
                           values_in.ravel(),
                           (lons_out, lats_out), method=interp,
                           fill_value=nodata)
    return values_out
import numpy as np

from pygeogrids.grids import BasicGrid
from os.path import join
from netCDF4 import Dataset

def ERA5025Cellgrid(path_grid, file_grid='grid.nc'):

    data_grid = Dataset(join(path_grid, file_grid))

    try:
        lon_1d = data_grid['longitude'][:]
    except BaseException:
        lon_1d = data_grid['Longitude'][:]
    try:
        lat_1d = data_grid['latitude'][:]
    except BaseException:
        lat_1d = data_grid['Latitude'][:]
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

    return BasicGrid(lon_2d.flatten(), lat_2d.flatten()).to_cell_grid(cellsize=5.)

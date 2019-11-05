import numpy as np

from pygeogrids.grids import BasicGrid
from os.path import join
from netCDF4 import Dataset

def RZSMCellgrid(path_grid, file_grid='grid.nc'):

    data_grid = Dataset(join(path_grid, file_grid))

    lon_1d = data_grid['lon'][:]
    lat_1d = data_grid['lat'][:]

    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

    return BasicGrid(lon_2d.flatten(), lat_2d.flatten()).to_cell_grid(cellsize=5.)

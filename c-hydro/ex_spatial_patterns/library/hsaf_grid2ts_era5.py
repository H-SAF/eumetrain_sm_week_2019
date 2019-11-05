# ----------------------------------------------------------------------------------------------------------------------
# Libraries
from os.path import join

from era5.reshuffle import main
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Define date(s) and variable(s)
date_from = '2007-01-01'
date_to = '2015-01-01'
var = ["tp", "skt"]

# Define base path
path_base = '/home/fabio/Desktop/PyCharm_Workspace_Python2/HSAF/Product_Validation_Analysis/download/'
# Define product folder(s)
folder_grid = 'era5_grid/morocco/hour/'
folder_ts = 'era5_ts/morocco/'
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Create path of grid and ts data
path_grid = join(path_base, folder_grid)
path_ts = join(path_base, folder_ts)

# Define args
args = [path_grid, path_ts, date_from, date_to] + var
# Call and run procedure to convert GLDAS Noah data from grid to ts
main(args)
# ----------------------------------------------------------------------------------------------------------------------

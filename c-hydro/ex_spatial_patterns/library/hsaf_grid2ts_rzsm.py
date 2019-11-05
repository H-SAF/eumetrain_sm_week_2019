# ----------------------------------------------------------------------------------------------------------------------
# Libraries
from os.path import join

from rzsm.reshuffle import main
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Define date(s) and variable(s)
date_from = '2015-01-01'
date_to = '2015-03-01'
var = ["var40", "var41", "var42", "var43"]

# Define base path
path_base = '/home/fabio/Desktop/PyCharm_Workspace_Python2/HSAF/Product_Validation_Analysis/download/'
# Define product folder(s)
folder_grid = 'rzsm_grid/italy/'
folder_ts = 'rzsm_ts/italy/'

folder_static = 'rzsm_grid/italy/static/'

domain = 'italy'
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Create path of grid and ts data
path_grid = join(path_base, folder_grid)
path_ts = join(path_base, folder_ts)

path_static = join(path_base, folder_static)

# Define args
args = [path_grid, path_ts, path_static, date_from, date_to, domain] + var
# Call and run procedure to convert GLDAS Noah data from grid to ts
main(args)
# ----------------------------------------------------------------------------------------------------------------------

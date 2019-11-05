# ----------------------------------------------------------------------------------------------------------------------
# Libraries
import os
from os.path import join, exists
import numpy as np
import pandas as pd
from netCDF4 import Dataset

try:
    from ascat.timeseries import AscatSsmCdr
except BaseException:
    from ascat import AscatSsmCdr

from library.era5.interface import ERA5Ts
from library.rzsm.interface import RZSMTs

from library.cima.ts_utils import get_bit
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Class to wrap ERA5 data record
class RZSM_Dataset_TS(RZSMTs):

    # ----------------------------------------------------------------------------------------------------------------------
    # Initialize class
    def __init__(self, *args, **kwargs):

        # initialise static layer paths
        if 'dr_path' in kwargs:
            self.dr_path = kwargs.pop('dr_path')
        else:
            raise ValueError('data record path not defined')

        if "temp_path" in kwargs:
            self.temp_path = kwargs.pop('temp_path')

            if not os.path.exists(self.temp_path):
                os.makedirs(self.temp_path)

        else:
            raise ValueError('temp path not defined')

        # initialise static layer paths
        if 'grid_path' in kwargs:
            self.grid_path = kwargs.pop('grid_path')
        else:
            raise ValueError('grid layer path not defined')

        super(RZSM_Dataset_TS, self).__init__(ts_path=self.dr_path, grid_path=self.grid_path,
                                              **kwargs)

    # ----------------------------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------------------------
    # Method to read time-series
    def read_ts(self, *args, **kwargs):
        """
        Method to read time series and mask the data
        """
        try:
            gpi_era = args[0]

            temp_file = join(self.temp_path, 'rzsm_ts_gpi_' + str(gpi_era) + '.pkl')
            if not exists(temp_file):
                ts = super(RZSM_Dataset_TS, self).read(*args, **kwargs)
                ts.to_pickle(temp_file)
            else:
                ts = pd.read_pickle(temp_file)

            #ts = ts * 100

            if ts.size == 0:
                print(' ----> WARNING: No data valid for RZSM dataset')
                ts = pd.DataFrame()

        except Exception as ex:

            message = (type(ex).__name__, ex.args)
            print(message)
            print(' ----> WARNING: RunTime error for RZSM dataset -- ' + repr(ex))
            ts = pd.DataFrame()

        return ts
    # ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# Class to wrap ERA5 data record
class ERA5_Dataset_TS(ERA5Ts):

    # ----------------------------------------------------------------------------------------------------------------------
    # Initialize class
    def __init__(self, *args, **kwargs):

        # initialise static layer paths
        if 'dr_path' in kwargs:
            self.dr_path = kwargs.pop('dr_path')
        else:
            raise ValueError('data record path not defined')

        if "temp_path" in kwargs:
            self.temp_path = kwargs.pop('temp_path')

            if not os.path.exists(self.temp_path):
                os.makedirs(self.temp_path)

        else:
            raise ValueError('temp path not defined')

        # initialise static layer paths
        if 'grid_path' in kwargs:
            self.grid_path = kwargs.pop('grid_path')
        else:
            raise ValueError('grid layer path not defined')

        super(ERA5_Dataset_TS, self).__init__(ts_path=self.dr_path, grid_path=self.grid_path,
                                              **kwargs)

    # ----------------------------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------------------------
    # Method to read time-series
    def read_ts(self, *args, **kwargs):
        """
        Method to read time series and mask the data
        """
        try:
            gpi_era = args[0]

            temp_file = join(self.temp_path, 'era5_ts_gpi_' + str(gpi_era) + '.pkl')
            if not exists(temp_file):
                ts = super(ERA5_Dataset_TS, self).read(*args, **kwargs)
                ts.to_pickle(temp_file)
            else:
                ts = pd.read_pickle(temp_file)

            ts = ts.astype(dtype={"skt": "float16", "tp": "float16"})
            ts['tp'] = ts['tp'] * 1000  # from m to mm of total precipitation

            if ts.size == 0:
                print(' ----> WARNING: No data valid for ERA5 dataset')
                ts = pd.DataFrame()

        except Exception as ex:

            message = (type(ex).__name__, ex.args)
            print(message)
            print(' ----> WARNING: RunTime error for ERA5 dataset -- ' + repr(ex))
            ts = pd.DataFrame()

        return ts
    # ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Class to wrap ASCAT data record
class ASCAT_Dataset_DR(AscatSsmCdr):

    # ----------------------------------------------------------------------------------------------------------------------
    # Initialize class variable(s)
    por = None
    # ----------------------------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------------------------
    # Initialize class
    def __init__(self, *args, **kwargs):

        # initialise static layer paths
        if 'dr_path' in kwargs:
            self.dr_path = kwargs.pop('dr_path')
        else:
            raise ValueError('data record path not defined')

        if "temp_path" in kwargs:
            self.temp_path = kwargs.pop('temp_path')

            if not os.path.exists(self.temp_path):
                os.makedirs(self.temp_path)

        else:
            raise ValueError('temp path not defined')

        # initialise static layer paths
        if 'grid_path' in kwargs:
            self.grid_path = kwargs.pop('grid_path')
        else:
            raise ValueError('grid layer path not defined')

        # initialise static layer paths
        if 'static_layer_path' in kwargs:
            self.static_layer_path = kwargs.pop('static_layer_path')
        else:
            raise ValueError('static layer path not defined')

        if 'file_porosity' in kwargs:
            self._por_file = kwargs.pop('file_porosity')
        else:
            self._por_file = 'porosity.nc'

        self._por_path = os.path.join(self.static_layer_path, self._por_file)

        #self._read_porosity()

        super(ASCAT_Dataset_DR, self).__init__(cdr_path=self.dr_path,
                                               grid_path=self.grid_path,
                                               grid_filename='TUW_WARP5_grid_info_2_2.nc',
                                               static_layer_path=self.static_layer_path,
                                               **kwargs)

    # ----------------------------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------------------------
    # Method to read porosity file
    def _read_porosity(self):
        """
        Read global porosity from NOAH GLDAS.
        """

        if self.por is None:

            ncFile = Dataset(self._por_path, mode='r')
            por_gpi = ncFile['location_id'][:]
            por = ncFile['por_gldas'][:]
            self.por = por[~por.mask]
            self.por_gpi = por_gpi[~por.mask]

            ncFile.close()

        # with Dataset(self._por_path, mode='r') as ncFile:
        #     por_gpi = ncFile['location_id'][:]
        #     por = ncFile['por_gldas'][:]
        #     self.por = por[~por.mask]
        #     self.por_gpi = por_gpi[~por.mask]
        #
        #     ncFile.close()

    # ----------------------------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------------------------
    # Method to get porosity value
    def get_porosity(self, *args):
        """
        Read porosity for given location.

        Takes either 1 or 2 arguments and calls the correct function
        which is either reading the gpi directly or finding
        the nearest gpi from given lat/lon coordinates and then reading it.

        Returns
        -------
        por : float32
            Porosity.
        """
        if len(args) == 1:
            gpi = args[0]
        if len(args) == 2:
            gpi, _ = self.grid.find_nearest_gpi(args[0], args[1])
        if len(args) < 1 or len(args) > 2:
            raise ValueError('Wrong number of arguments.')

        ind = np.where(self.por_gpi == gpi)[0]

        if ind.size == 0:
            por = np.nan
        else:
            por = self.por[ind]

        return por

    # ----------------------------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------------------------
    # Method to read time-series
    def read_ts(self, *args, **kwargs):
        """
        Method to read time series and mask the data
        """
        try:
            gpi_ascat = args[0]

            temp_file = join(self.temp_path, 'ascat_ts_gpi_' + str(gpi_ascat) + '.pkl')
            if not exists(temp_file):
                ts_obj = super(ASCAT_Dataset_DR, self).read(*args, **kwargs)
                ts = ts_obj.data
                ts.to_pickle(temp_file)
            else:
                ts = pd.read_pickle(temp_file)

            ts = ts.astype(dtype={"sm": "float64"})

            bit_mask = ((get_bit(ts['proc_flag'], 1)) |
                        (get_bit(ts['proc_flag'], 2)) |
                        (get_bit(ts['proc_flag'], 3)))

            ts = ts[((ts['ssf'] == 0) | (ts['ssf'] == 1)) & (bit_mask == 0)]

            ts['sm'] = ts['sm'] / 100

            '''
            # convert to absolute soil moisture
            porosity = self.get_porosity(*args)

            if porosity is not np.nan:
                ts['sm'] = ts['sm'] / 100. * porosity
                ts['sm_noise'] = ts['sm_noise'] / 100. * porosity
            else:
                print(' ----> WARNING: No porosity valid for ASCAT dataset')
                ts = pd.DataFrame()
            
            '''
            if ts.size == 0:
                print(' ----> WARNING: No data valid for ASCAT dataset')
                ts = pd.DataFrame()

        except Exception as ex:

            message = (type(ex).__name__, ex.args)
            print(message)
            print(' ----> WARNING: RunTime error for ASCAT dataset -- ' + repr(ex))
            ts = pd.DataFrame()

        return ts

    # ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------

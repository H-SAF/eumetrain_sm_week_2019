# -*- coding: utf-8 -*-

import warnings
import numpy as np
import os

try:
    import pygrib
except ImportError:
    warnings.warn("pygrib has not been imported")

from pygeobase.io_base import ImageBase, MultiTemporalImageBase
from pygeobase.object_base import Image
import pygeogrids
from pygeogrids import netcdf
from pynetcf.time_series import GriddedNcOrthoMultiTs

from datetime import timedelta

from library.era5.grid import ERA5025Cellgrid
from netCDF4 import Dataset

import matplotlib.pylab as plt

class ECMWF_ERA5_025Img(ImageBase):
    """
    Class for reading one GLDAS Noah v2.1 nc file in 0.25° grid.

    Parameters
    ----------
    filename: string
        filename of the GLDAS nc file
    mode: string, optional
        mode of opening the file, only 'r' is implemented at the moment
    parameter : string or list, optional
        one or list of parameters to read, see GLDAS v2.1 documentation for more information
        Default : 'SoilMoi0_10cm_inst'
    array_1D: boolean, optional
        if set then the data is read into 1D arrays. Needed for some legacy code.
    """
    
    def __init__(self, filename, mode='r', parameter='skt', array_1D=False, path_grid=None):
        super(ECMWF_ERA5_025Img, self).__init__(filename, mode=mode)

        if type(parameter) != list:
            parameter = [parameter]
        self.path_grid = path_grid
        self.parameters = parameter
        self.fill_values = np.repeat(9999., 0)
        self.grid = ERA5025Cellgrid(self.path_grid)
        self.array_1D = array_1D
    
    def read(self, timestamp=None):
        
        #print 'read file: %s' %self.filename
        #Returns the selected parameters for a gldas image and according metadata
        return_img = {}
        return_metadata = {}
        
        try:
            dataset = Dataset(self.filename)
        except IOError as e:
            print(e)
            print(" ".join([self.filename, "can not be opened"]))
            raise e
         

        param_names=[]
        for parameter in self.parameters:
            param_names.append(parameter)
            
        
        for parameter, variable in dataset.variables.items():
            if parameter in param_names:
                param_metadata={}
                param_data={}           
                for attrname in variable.ncattrs():
                    if attrname in ['long_name', 'units']:
                        param_metadata.update({str(attrname):getattr(variable,attrname)})

                param_data = dataset.variables[parameter][:]
                param_data = param_data[0, :, :]
                np.ma.set_fill_value(param_data, 9999)
                
                #param_data=np.concatenate((self.fill_values,
                #                           np.ma.getdata(param_data.filled())
                #                           .flatten()))

                #plt.figure(1)
                #plt.imshow(param_data)
                #plt.show()

                param_data=np.concatenate((self.fill_values,
                                           param_data.flatten()))



                return_img.update({str(parameter): param_data[self.grid.activegpis]})
                return_metadata.update({str(parameter): param_metadata})
                        
            #Check for corrupt files
                try:
                    return_img[parameter]
                except KeyError:
                    path, thefile = os.path.split(self.filename)
                    print ('%s in %s is corrupt - filling image with NaN values' %(parameter,thefile))
                    return_img[parameter]  = np.empty(self.grid.n_gpi).fill(np.nan)
                    return_metadata['corrupt_parameters'].append()
                    
        dataset.close()
        if self.array_1D:
            return Image(self.grid.activearrlon,
                         self.grid.activearrlat,
                         return_img,
                         return_metadata,
                         timestamp)
        else:
            for key in return_img:
                return_img[key]=np.flipud(return_img[key].reshape((720,1440)))
            
            return Image(np.flipud(self.grid.activearrlon.reshape((720,1440))),
                         np.flipud(self.grid.activearrlat.reshape((720,1440))),
                         return_img, 
                         return_metadata,
                         timestamp)
                        
                
    def write(self,data):
        raise NotImplementedError()
        
    def flush(self):
        pass
    
    def close(self):
        pass


class ECMWF_ERA5_025Ds(MultiTemporalImageBase):
    """
    Class for reading GLDAS v2.1 images in nc format.

    Parameters
    ----------
    data_path : string
        path to the nc files
    parameter : string or list, optional
        one or list of parameters to read, see GLDAS v2.1 documentation for more information
        Default : 'SoilMoi0_10cm_inst'
    array_1D: boolean, optional
        if set then the data is read into 1D arrays. Needed for some legacy code.
    """

    def __init__(self, data_path, parameter='skt', array_1D=False):

        ioclass_kws = {'parameter': parameter,
                       'array_1D': array_1D,
                       'path_grid': data_path}

        sub_path = ['%Y', "%m", "%d"]

        filename_templ = "era5_{datetime}.nc"
        #filename_templ = "GLDAS_NOAH025_3H.A{datetime}.*.nc4"
        super(ECMWF_ERA5_025Ds, self).__init__(data_path, ECMWF_ERA5_025Img,
                                                  fname_templ=filename_templ,
                                                  datetime_format="%Y%m%d%H%M",
                                                  subpath_templ=sub_path,
                                                  exact_templ=False,
                                                  ioclass_kws=ioclass_kws)

    def tstamps_for_daterange(self, start_date, end_date):
        """
        return timestamps for daterange,

        Parameters
        ----------
        start_date: datetime
            start of date range
        end_date: datetime
            end of date range

        Returns
        -------
        timestamps : list
            list of datetime objects of each available image between
            start_date and end_date
        """
        img_offsets = np.array([timedelta(hours=0),
                                timedelta(hours=1),
                                timedelta(hours=2),
                                timedelta(hours=3),
                                timedelta(hours=4),
                                timedelta(hours=5),
                                timedelta(hours=6),
                                timedelta(hours=7),
                                timedelta(hours=8),
                                timedelta(hours=9),
                                timedelta(hours=10),
                                timedelta(hours=11),
                                timedelta(hours=12),
                                timedelta(hours=13),
                                timedelta(hours=14),
                                timedelta(hours=15),
                                timedelta(hours=16),
                                timedelta(hours=17),
                                timedelta(hours=18),
                                timedelta(hours=19),
                                timedelta(hours=20),
                                timedelta(hours=21),
                                timedelta(hours=22),
                                timedelta(hours=23)])

        timestamps = []
        diff = end_date - start_date
        for i in range(diff.days + 1):
            daily_dates = start_date + timedelta(days=i) + img_offsets
            timestamps.extend(daily_dates.tolist())

        return timestamps

class ERA5Ts(GriddedNcOrthoMultiTs):

    def __init__(self, ts_path, grid_path=None):

        if grid_path is None:
            grid_path = os.path.join(ts_path, "grid.nc")
        else:
            grid_path = os.path.join(grid_path, "grid.nc")

        grid = netcdf.load_grid(grid_path)
        super(ERA5Ts, self).__init__(ts_path, grid)

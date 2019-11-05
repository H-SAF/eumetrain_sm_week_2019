# -*- coding: utf-8 -*-
# The MIT License (MIT)
#
# Copyright (c) 2016, TU Wien
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''
Module for a command line interface to convert the GLDAS data into a
time series format using the repurpose package
'''

import numpy as np
import os
import sys
import argparse
from datetime import datetime

from pygeogrids import BasicGrid

from repurpose.img2ts import Img2Ts
from rzsm.interface import ECMWF_RZSM_Ds

import matplotlib.pylab as plt


def get_filetype(inpath):
    '''
    Tries to find out the file type by searching for
    grib or nc files two subdirectories into the passed input path.
    If function fails, grib is assumed.

    Parameters
    ------------
    input_root: string
        input path where GLDAS data was downloaded
    '''

    onedown = os.path.join(inpath, os.listdir(inpath)[0])
    twodown = os.path.join(onedown, os.listdir(onedown)[0])

    filelist = []
    for path, subdirs, files in os.walk(twodown):
        for name in files:
            filename, extension = os.path.splitext(name)
            filelist.append(extension)

    if '.nc4' in filelist and '.grb' not in filelist:
        return 'netCDF'
    elif '.grb' in filelist and '.nc4' not in filelist:
        return 'grib'
    else:
        # if file type cannot be detected, guess grib
        return 'grib'


def mkdate(datestring):
    if len(datestring) == 10:
        return datetime.strptime(datestring, '%Y-%m-%d')
    if len(datestring) == 16:
        return datetime.strptime(datestring, '%Y-%m-%dT%H:%M')


def reshuffle(input_root, outputpath, static_root,
              startdate, enddate, domain,
              parameters,
              imgbuffer=50):
    """
    Reshuffle method applied to ERA-Interim data.

    Parameters
    ----------
    input_root: string
        input path where era interim data was downloaded
    outputpath : string
        Output path.
    startdate : datetime
        Start date.
    enddate : datetime
        End date.
    parameters: list
        parameters to read and convert
    imgbuffer: int, optional
        How many images to read at once before writing time series.
    """

    input_dataset = ECMWF_RZSM_Ds(input_root, parameters, array_1D=True, path_grid=static_root, domain=domain)

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    global_attr = {'product': 'ZRSM'}

    # get time series attributes from first day of data.
    data = input_dataset.read(startdate)
    ts_attributes = data.metadata
    grid = BasicGrid(data.lon, data.lat)

    # test
    #test_data = data['skt']
    #test_data_res = np.reshape(test_data, (720,1440))
    #test_lon_res = np.reshape(data.lon, (720, 1440))
    #test_lat_res = np.reshape(data.lat, (720, 1440))

    #test_data_res[test_data_res > 100] = np.nan

    #plt.figure(1)
    #plt.pcolor(test_lon_res, test_lat_res, test_data_res)
    #plt.show()

    reshuffler = Img2Ts(input_dataset=input_dataset, outputpath=outputpath,
                        startdate=startdate, enddate=enddate,
                        input_grid=grid,
                        imgbuffer=imgbuffer, cellsize_lat=5.0, cellsize_lon=5.0,
                        global_attr=global_attr,
                        zlib=True,
                        unlim_chunksize=1000,
                        ts_attributes=ts_attributes)
    reshuffler.calc()


def parse_args(args):
    """
    Parse command line parameters for conversion from image to timeseries

    :param args: command line parameters as list of strings
    :return: command line parameters as :obj:`argparse.Namespace`
    """
    parser = argparse.ArgumentParser(
        description="Convert GLDAS data to time series format.")
    parser.add_argument("dataset_root",
                        help='Root of local filesystem where the data is stored.')
    parser.add_argument("timeseries_root",
                        help='Root of local filesystem where the timeseries should be stored.')
    parser.add_argument("static_root",
                        help='')
    parser.add_argument("start", type=mkdate,
                        help=("Startdate. Either in format YYYY-MM-DD or YYYY-MM-DDTHH:MM."))
    parser.add_argument("end", type=mkdate,
                        help=("Enddate. Either in format YYYY-MM-DD or YYYY-MM-DDTHH:MM."))
    parser.add_argument("domain", metavar="domain",help="")
    parser.add_argument("parameters", metavar="parameters",
                        nargs="+",
                        help=("Parameters to download in numerical format. e.g."
                              "086_L1 086_L2 086_L3 086_L4 for Volumetric soil water layers 1 to 4."))

    parser.add_argument("--imgbuffer", type=int, default=50,
                        help=("How many images to read at once. Bigger numbers make the "
                              "conversion faster but consume more memory."))
    args = parser.parse_args(args)
    # set defaults that can not be handled by argparse

    print("Converting data from {} to {} into folder {}.".format(args.start.isoformat(),
                                                                 args.end.isoformat(),
                                                                 args.timeseries_root))
    return args


def main(args):
    args = parse_args(args)

    reshuffle(args.dataset_root,
              args.timeseries_root,
              args.static_root,
              args.start,
              args.end,
              args.domain,
              args.parameters,
              imgbuffer=args.imgbuffer)


def run():
    main(sys.argv[1:])

if __name__ == '__main__':
    run()

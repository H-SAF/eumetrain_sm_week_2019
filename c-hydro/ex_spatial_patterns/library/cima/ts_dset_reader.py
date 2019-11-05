from datetime import datetime
from library.cima.ts_dset_driver import ASCAT_Dataset_DR, ERA5_Dataset_TS, RZSM_Dataset_TS

def dset_init(settings):

    reader_ascat = ASCAT_Dataset_DR(dr_path=settings['ascat_path_ts'],
                                    temp_path=settings['ascat_path_tmp'],
                                    grid_path=settings['ascat_path_grid'],
                                    static_layer_path=settings['ascat_path_layer'])

    reader_era5 = ERA5_Dataset_TS(dr_path=settings['era5_path_ts'],
                                  temp_path=settings['era5_path_tmp'],
                                  grid_path=settings['era5_path_grid'])

    reader_rzsm = RZSM_Dataset_TS(dr_path=settings['rzsm_path_ts'],
                                  temp_path=settings['rzsm_path_tmp'],
                                  grid_path=settings['rzsm_path_grid'])

    return reader_ascat, reader_era5, reader_rzsm

def dset_config(reader_ascat, reader_era5, reader_rzsm, settings):

    datasets = {
        'ASCAT': {
            'class': reader_ascat,
            'columns': ['sm'],
            'type': 'reference',
            'args': [],
            'kwargs': {'mask_frozen_prob': settings['ascat_mask_frozen_prob_threshold'],
                       'mask_snow_prob': settings['ascat_mask_snow_prob_threshold']}
        },
        'ERA5': {
            'class': reader_era5,
            'columns': ['tp', 'tsk'],
            'type': 'other',
            'grids_compatible': False,
            'use_lut': True,
            'lut_max_dist': settings['max_dist'],
        },
        'RZSM': {
            'class': reader_rzsm,
            'columns': ["var40", "var41", "var42", "var43"],
            'type': 'other',
            'grids_compatible': False,
            'use_lut': True,
            'lut_max_dist': settings['max_dist'],
        },
    }

    return datasets

def dset_period(settings):
    period = [datetime.strptime(settings['time_start'], settings['time_format']),
              datetime.strptime(settings['time_end'], settings['time_format'])]
    return period

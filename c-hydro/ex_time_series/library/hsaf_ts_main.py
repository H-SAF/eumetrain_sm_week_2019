

# Libraries
import numpy as np
import pandas as pd
from os.path import join

from hsaf_domain_utils import get_grid, get_file_shp, get_file_json, create_points_shp
from pytesmo.scaling import get_scaling_function, get_scaling_method_lut
from pytesmo.time_series.filters import exp_filter
from pytesmo.time_series import anomaly
from hsaf_ts_utils import df_time_matching, df_temporal_matching, df_period_selection
from hsaf_map_utils import interpolate_point2map, create_map, create_image
from hsaf_ts_dset_reader import dset_init, dset_config, dset_period

from library.irpi.indices.drought import ssi, spi

import matplotlib.pyplot as plt

#from library.irpi.indices.ssi import SSIcal
#from library.irpi.indices.spi import SPIcal

# Information
sFileName_BASIN_SHP = 'tiber_basin.shp'
sFileName_BASIN_ERA5_SHP = 'tiber_basin_era5.shp'
sFileName_BASIN_ASCAT_SHP = 'tiber_basin_ascat.shp'
sFileName_BASIN_RZSM_SHP = 'tiber_basin_rzsm.shp'

sFileName_ASCAT_SHP = 'tiber_ascat.shp'
sFileName_ERA5_SHP = 'tiber_era5.shp'
sFileName_RZSM_SHP = 'tiber_rzsm.shp'

sFileName_BASIN_TIFF = 'tiber_basin_mask_bbox.tiff'
sFolderName_Land = '/home/fabio/Desktop/PyCharm_Workspace_Python2/HSAF/Training_Course_2018/data/static/shp/tiber_basin/'

sFolderName_Alg = '/home/fabio/Desktop/PyCharm_Workspace_Python2/HSAF/Training_Course_2018/'
sFileName_Alg = 'configuration_datasets_italy.json'

sFolderName_Results = '/home/fabio/Desktop/PyCharm_Workspace_Python2/HSAF/Training_Course_2018/data/results/'
sFileName_ERA5_SKT_TIFF = 'tiber_basin_era5_skt.tiff'

time_step = "2007-03-05 15:00"

# ----------------------------------------------------------------------------------------------------------------------
# Get configuration info
settings = get_file_json(join(sFolderName_Alg, sFileName_Alg))
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Get land info (using shp file)
basin_rows, basin_cols, basin_epsg, basin_transform, basin_meta_reference = get_file_shp(
    join(sFolderName_Land, sFileName_BASIN_SHP), join(sFolderName_Land, sFileName_BASIN_TIFF),
    cell_size=0.005, bbox_ext=0)

# Create basin grid
basin_grid, basin_lons_2d, basin_lats_2d, basin_bbox = get_grid(join(sFolderName_Land, sFileName_BASIN_TIFF))
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Initialize ascat and era dataset(s)
reader_ascat, reader_era5, reader_rzsm = dset_init(settings)
datasets = dset_config(reader_ascat, reader_era5, reader_rzsm, settings)
period = dset_period(settings)

# Get scaling method
scaling_methods = get_scaling_method_lut()

scaling_method_lr = get_scaling_function('linreg')
scaling_method_ms = get_scaling_function('mean_std')
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Get ascat gpi(s)
gpis_ascat, lats_ascat, lons_ascat = reader_ascat.grid.get_bbox_grid_points(
    latmin=basin_bbox.bottom, latmax=basin_bbox.top, lonmin=basin_bbox.left,
    lonmax=basin_bbox.right, both=True)
# Get era5 gpi(s)
gpis_era5, lats_era5, lons_era5 = reader_era5.grid.get_bbox_grid_points(
    latmin=basin_bbox.bottom, latmax=basin_bbox.top, lonmin=basin_bbox.left,
    lonmax=basin_bbox.right, both=True)
# Get rzsm gpi(s)
gpis_rzsm, lats_rzsm, lons_rzsm = reader_rzsm.grid.get_bbox_grid_points(
    latmin=basin_bbox.bottom, latmax=basin_bbox.top, lonmin=basin_bbox.left,
    lonmax=basin_bbox.right, both=True)

gpis_ascat_n = gpis_ascat.__len__()
gpis_era5_n = gpis_era5.__len__()
gpis_rzsm_n = gpis_rzsm.__len__()
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Get era5 gpi using ascat reference grid
luts_ascat_era5 = reader_era5.grid.calc_lut(reader_ascat.grid, max_dist=settings['max_dist'])
gpis_ascat_era5 = np.unique(luts_ascat_era5[gpis_era5])
lons_ascat_era5, lats_ascat_era5 = reader_ascat.grid.gpi2lonlat(gpis_ascat_era5)
# Get rzsm gpi using ascat reference grid
luts_ascat_rzsm = reader_rzsm.grid.calc_lut(reader_ascat.grid, max_dist=settings['max_dist'])
gpis_ascat_rzsm = np.unique(luts_ascat_rzsm[gpis_rzsm])
lons_ascat_rzsm, lats_ascat_rzsm = reader_ascat.grid.gpi2lonlat(gpis_ascat_rzsm)

luts_ascat = reader_ascat.grid.calc_lut(basin_grid, max_dist=settings['max_dist'])
gpis_basin_ascat = np.unique(luts_ascat[gpis_ascat])
lons_basin_ascat, lats_basin_ascat = basin_grid.gpi2lonlat(gpis_basin_ascat)

gpis_ascat = basin_grid.calc_lut(reader_ascat.grid, max_dist=settings['max_dist'])
gpis_ascat = np.unique(gpis_ascat)
lons_ascat, lats_ascat = reader_ascat.grid.gpi2lonlat(gpis_ascat)

luts_era5 = reader_era5.grid.calc_lut(basin_grid, max_dist=settings['max_dist'])
gpis_basin_era5 = np.unique(luts_era5[gpis_era5])
lons_basin_era5, lats_basin_era5 = basin_grid.gpi2lonlat(gpis_basin_era5)

luts_rzsm = reader_rzsm.grid.calc_lut(basin_grid, max_dist=settings['max_dist'])
gpis_basin_rzsm = np.unique(luts_rzsm[gpis_rzsm])
lons_basin_rzsm, lats_basin_rzsm = basin_grid.gpi2lonlat(gpis_basin_rzsm)

# Get era5 gpi using rzsm reference grid
luts_rzsm_era5 = reader_era5.grid.calc_lut(reader_rzsm.grid, max_dist=settings['max_dist'])
gpis_rzsm_era5 = np.unique(luts_rzsm_era5[gpis_era5])
lons_rzsm_era5, lats_rzsm_era5 = reader_rzsm.grid.gpi2lonlat(gpis_rzsm_era5)

# Define ascat, era5 and rzsm common gpis
gpis_ascat = reader_ascat.grid.find_nearest_gpi(lons_rzsm_era5, lats_rzsm_era5, max_dist=settings['max_dist'])
lons_ascat, lats_ascat = reader_ascat.grid.gpi2lonlat(gpis_ascat[0])
gpis_era5 = reader_era5.grid.find_nearest_gpi(lons_ascat, lats_ascat, max_dist=settings['max_dist'])
lons_era5, lats_era5 = reader_era5.grid.gpi2lonlat(gpis_era5[0])
gpis_rzsm = reader_rzsm.grid.find_nearest_gpi(lons_ascat, lats_ascat, max_dist=settings['max_dist'])
lons_rzsm, lats_rzsm = reader_rzsm.grid.gpi2lonlat(gpis_rzsm[0])
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Create shape files to check gpi(s)
create_points_shp(gpis_ascat[0], lons_ascat, lats_ascat,
                  file_name_shp=join(sFolderName_Land, sFileName_ASCAT_SHP))
create_points_shp(gpis_era5[0], lons_era5, lats_era5,
                  file_name_shp=join(sFolderName_Land, sFileName_ERA5_SHP))
create_points_shp(gpis_rzsm[0], lons_rzsm, lats_rzsm,
                  file_name_shp=join(sFolderName_Land, sFileName_RZSM_SHP))

create_points_shp(gpis_basin_era5, lons_basin_era5, lats_basin_era5,
                  file_name_shp=join(sFolderName_Land, sFileName_BASIN_ERA5_SHP))
create_points_shp(gpis_basin_ascat, lons_basin_ascat, lats_basin_ascat,
                  file_name_shp=join(sFolderName_Land, sFileName_BASIN_ASCAT_SHP))
create_points_shp(gpis_basin_rzsm, lons_basin_rzsm, lats_basin_rzsm,
                  file_name_shp=join(sFolderName_Land, sFileName_BASIN_RZSM_SHP))
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Iterate over ascat gpis
for gpi_ascat, gpi_era5, gpi_rzsm in zip(gpis_ascat[0], gpis_era5[0], gpis_rzsm[0]):

    # Get time-series data
    ts_ascat = reader_ascat.read_ts(gpi_ascat)
    ts_era5 = reader_era5.read_ts(gpi_era5)
    ts_rzsm = reader_rzsm.read_ts(gpi_rzsm)
    # Select time-series period
    ts_ascat = ts_ascat.loc[settings['time_start']:settings['time_end']]
    ts_era5 = ts_era5.loc[settings['time_start']:settings['time_end']]
    ts_rzsm = ts_rzsm.loc[settings['time_start']:settings['time_end']]

    # Resample time-series to daily values
    ts_resample = pd.DataFrame()
    ts_resample['sm'] = ts_ascat.sm.resample('D').mean().dropna()
    ts_resample['var40'] = ts_rzsm.var40.resample('D').mean().dropna()
    ts_resample['skt'] = ts_era5.skt.resample('D').mean().dropna()
    ts_resample['tp'] = ts_era5.tp.resample('D').sum().dropna()

    ts_resample['ssi'] = ssi(ts_resample, 1, 'sm')
    ts_resample['spi'] = spi(ts_resample, 1, 'tp')

    # Scale soil moisture (sm, var40)
    ts_ascat_scaled = scaling_method_ms(ts_resample.sm, ts_resample.var40)
    ts_resample['sm_scaled'] = ts_ascat_scaled

    # Match time-series (drop duplicates set to false)
    ts_matched = pd.DataFrame()
    ts_era5_matched, ts_ascat_matched = df_temporal_matching(ts_era5, ts_ascat,
                                                             name_ref='ERA5', name_k='ASCAT',
                                                             window=settings["temporal_matching"],
                                                             drop_duplicates=settings["temporal_drop_duplicates"])
    ts_matched['skt'] = ts_era5_matched['skt']
    ts_matched['tp'] = ts_era5_matched['tp']
    ts_matched['sm'] = ts_ascat_matched['sm']

    ts_matched.plot()

    # Plot skt and tp variable(s) in same panel with two graph
    fig, axs = plt.subplots(2, 1, figsize=(17, 11))
    axs[0].plot(ts_resample['skt'], color='r')
    axs[1].bar(ts_resample['tp'].index, ts_resample['tp'].values, color='#0000FF', alpha=0.35, width=2, align='edge')

    axs[0].set_title('ERA5 Time-Series')
    axs[0].set_ylabel('Skin Temperature [K]', color='#BA3723')
    axs[1].set_ylabel('Total Precipitation [mm]', color='#0000FF')

    # compute average and sum daily, monthly, yearly value(s) [D, M, Y]
    ts_ascat_resample = pd.DataFrame()
    ts_ascat_resample['sm_avg'] = ts_ascat.sm.resample('D').mean()
    ts_ascat_resample['sm_sum'] = ts_ascat.sm.resample('D').sum()

    fig1, ax = plt.subplots(1, 1, figsize=(15, 5))
    ts_ascat['sm'].plot(ax=ax, lw=2, marker='o', label='sm')
    ts_ascat_resample['sm_avg'].plot(ax=ax, lw=2, marker='*', label='sm avg')
    plt.legend()

    ts_ascat_select = df_period_selection(ts_ascat, settings["time_analysis_start"], settings["time_analysis_end"])
    ts_ascat_select['sm'].plot()

    ts_era5_matched, ts_ascat_matched = df_temporal_matching(ts_era, ts_ascat,
                                                             name_ref='ERA5', name_k='ASCAT',
                                                             window=settings["temporal_matching"],
                                                             drop_duplicates=settings["temporal_drop_duplicates"])

    ts_era_value, ts_era_period = df_time_matching(ts_era, "2007-01-26 00:00", window=settings["temporal_matching"])
    ts_ascat_value, ts_ascat_period = df_time_matching(ts_ascat, "2007-01-26 00:00", window=settings["temporal_matching"])



    fig1, ax = plt.subplots(1, 1, figsize=(15, 5))
    ts_era5_matched['skt'].plot(ax=ax, lw=2, label='Skin Temperature [K]')
    ts_ascat_matched['sm'].plot(ax=ax, lw=2, label='sm')
    plt.legend()

    fig1, ax = plt.subplots(1, 1, figsize=(15, 5))
    ts_era['skt'].plot(ax=ax, lw=2, label='Skin Temperature [K]')
    plt.legend()

    fig2, ax = plt.subplots(1, 1, figsize=(15, 5))
    ts_era['skt'].plot.bar(ax=ax, label='Skin Temperature [K]')
    plt.legend()

    # Get ascat time-series

    ts_ascat = ts_ascat[['sm', 'sm_noise', 'frozen_prob', 'snow_prob', 'ssf', 'proc_flag']]
    ts_ascat.plot()
    ts_ascat['sm'].plot()

    # Drop NA measurements
    ts_ascat_sm = ts_ascat[['sm', 'sm_noise']].dropna()

    # Get julian dates of time series
    jd = ts_ascat_sm.index.to_julian_date().get_values()

    # Calculate SWI T=10
    ts_ascat_sm['swi_t10'] = exp_filter(ts_ascat_sm['sm'].values, jd, ctime=10)
    ts_ascat_sm['swi_t50'] = exp_filter(ts_ascat_sm['sm'].values, jd, ctime=50)

    fig1, ax = plt.subplots(1, 1, figsize=(15, 5))
    ts_ascat_sm['sm'].plot(ax=ax, alpha=0.4, marker='o', color='#00bfff', label='SSM')
    ts_ascat_sm['swi_t10'].plot(ax=ax, lw=2, label='SWI T=10')
    ts_ascat_sm['swi_t50'].plot(ax=ax, lw=2, label='SWI T=50')
    plt.legend()

    # Calculate anomaly based on moving +- 17 day window
    anomaly_ascat = anomaly.calc_anomaly(ts_ascat['sm'], window_size=35)
    fig2, ax = plt.subplots(1, 1, figsize=(15, 5))
    anomaly_ascat.plot(ax=ax, lw=2, label='ASCAT SM Anomaly')
    plt.legend()

    # Calculate climatology
    ts_ascat_clim = ts_ascat.dropna()
    climatology_ascat = anomaly.calc_climatology(ts_ascat_clim['sm'])
    fig3, ax = plt.subplots(1, 1, figsize=(15, 5))
    climatology_ascat.plot(ax=ax, lw=2, label='ASCAT SM Climatology')
    plt.legend()

    # Calculate anomaly based on climatology
    ts_ascat_clim = ts_ascat.dropna()
    anomaly_clim_ascat = anomaly.calc_anomaly(ts_ascat_clim['sm'], climatology=climatology_ascat)
    fig4, ax = plt.subplots(1, 1, figsize=(15, 5))
    anomaly_clim_ascat.plot(ax=ax, lw=2, label='ASCAT SM Anomaly vs Climatology')
    plt.legend()


    print('ciao')


# ----------------------------------------------------------------------------------------------------------------------
df_grid = pd.DataFrame(columns=['lon', 'lat', 'gpi', 'value', 'time'])
for gpi_era5, lon_era5, lat_era5 in zip(gpis_era5, lons_era5, lats_era5):
    ts_era5 = reader_era5.read_ts(gpi_era5)
    ts_era5_value, ts_era5_period = df_time_matching(ts_era5, time_step, window=settings["temporal_matching"])

    value_era5 = ts_era5_value['skt']
    time_era5 = ts_era5_value.name

    df_grid = df_grid.append({'lon': lon_era5, 'lat': lat_era5, 'gpi': gpi_era5,
                              'value': value_era5, 'time': time_era5}, ignore_index=True)

basin_grid_era5 = interpolate_point2grid(
    df_grid['lon'].values, df_grid['lat'].values, df_grid['value'].values, basin_lons_2d, basin_lats_2d)

create_map(basin_grid_era5, basin_rows, basin_cols, basin_epsg, basin_transform,
           file_name_data=join(sFolderName_Results, sFileName_ERA5_SKT_TIFF),
           file_name_mask=join(sFolderName_Land, sFileName_BASIN_TIFF))
# ----------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import os
import pandas as pd
import xarray as xr
from icecream import ic
from itertools import product
from shapely.geometry import Point, Polygon

var_map = {
    'tasmin': 'TMIN',
    'tasmax': 'TMAX'
}

def plot_climatology(ds, urban_vicinity, variable, URBAN, ucdb_city = [], 
                     valid_stations = [], time_series = [], city = None,
                     alpha_urb_borders = 1, 
                     linewidth_urb_borders = 2):
    """
    Plot the climatological data.

    Parameters:
        ds (xr.Dataset): Dataset containing the climatological data.
        ucdb_city (gpd.GeoDataFrame): GeoDataFrame of the city boundaries.
        urban_vicinity (object): Object representing urban vicinity.
        obs (pd.DataFrame, optional): DataFrame containing observational data (default is None).

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(12, 6))
                         
    # calculate climatology and anomaly from the model
    ds_var_period_mean = ds.mean('time').compute()
    rural_mean = ds_var_period_mean[variable].where(
        urban_vicinity['urmask'] == 0).mean().compute()
    ds_anomaly = ds_var_period_mean[variable] - rural_mean
    # Compute the maximum absolute value
    max_abs_value = abs(ds_anomaly).max().item()
                         
    if not isinstance(valid_stations, list):
        # calculate climatology and anomaly from the series
        codes_ins_city = valid_stations.code[valid_stations['inside_city'] == True]
        codes_out_city = valid_stations.code[valid_stations['inside_city'] == False]
        time_series_rural_mean = pd.DataFrame(index = time_series.index)
        time_series_rural_mean['rural_mean'] = time_series[codes_out_city].mean(axis = 1).values
        time_series_anomaly = time_series.sub(time_series_rural_mean['rural_mean'], axis = 0)

        ax.scatter(valid_stations.lon, valid_stations.lat, 
                   c = time_series_anomaly.mean(axis = 0, skipna=True), 
                   marker='o', cmap='bwr', 
                   s = 40, edgecolors = 'gray', 
                   vmin = -0.003, vmax = 0.003,
                   zorder = 10000) 

    im1 = ax.pcolormesh(ds.lon, ds.lat, ds_anomaly.values,
                    cmap='bwr', alpha = 0.7,
                    vmin = -0.003, vmax = 0.003)
    
    cbar = fig.colorbar(im1, ax = ax)    
    if not isinstance(valid_stations, list):#change
        ucdb_city.plot(ax=ax, facecolor="none", transform=proj, edgecolor="Green", linewidth=2, zorder = 1000)
    
    ax.coastlines()
    # Dynamically get the unit from the dataset
    unit = ds[variable].attrs.get('units', 'unknown')  # Default to 'unknown' if 'units' is missing

    cbar.set_label(f"{unit}", rotation = 90, fontsize = 14)
    if variable == 'tasmin':
        ax.set_title(f"Minimum temperature anomaly for {city}", fontsize=18)
        ax.set_ylabel(f"Minimum temperature anomaly ({unit})", fontsize=18)
    elif variable == 'tasmax':
        ax.set_title(f"Maximum temperature anomaly for {city}", fontsize=18)
        ax.set_ylabel(f"Maximum temperature anomaly ({unit})", fontsize=18)
    elif variable == 'huss':
        ax.set_title(f"Atmospheric moisture anomaly for {city}", fontsize=18)
        ax.set_ylabel(f"Atmospheric moisture anomaly ({unit})", fontsize=18)
    elif variable == 'hurs':
        ax.set_title(f"Relative atmospheric humidity anomaly for {city}", fontsize=18)
        ax.set_ylabel(f"Relative atmospheric humidity ({unit})", fontsize=18)
    elif variable == 'sfcWind':
        ax.set_title(f"Wind speed anomaly for {city}", fontsize=18)
        ax.set_ylabel(f"Wind speed anomaly ({unit})", fontsize=18)
    elif variable == 'pr':
        ax.set_title(f"Precipitation anomaly for {city}", fontsize=18)
        ax.set_ylabel(f"Precipitation anomaly ({unit})", fontsize=18)

    
    # Overlay the cell borders and handle NaNs
    #URBAN.plot_urban_borders(urban_vicinity, ax, 
    #                         alpha_urb_borders, 
    #                         linewidth_urb_borders)
    URBAN.plot_urban_polygon(urban_vicinity, ax)
    
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    return fig

def plot_time_series(ds_var, variable, urban_vicinity, 
                     time_series = [], valid_stations = [], 
                     data_squares = False, percentile = 100, 
                     var_map = var_map, ucdb_city = None, 
                     city = None, cache = ''):
    '''
    Plot time series data with optional urban area overlay and additional time series overlay.

    Parameters:
    ds_var (xarray.Dataset): Dataset containing the variable data.
    variable (str): Name of the variable of interest.
    urban_vicinity (xarray.Dataset): Dataset containing information about urban areas.
    time_series (list of pandas.DataFrame, optional): List of time series dataframes to overlay on the plot.
    data_squares (bool, optional): Flag indicating whether to plot individual data squares for urban and rural areas.

    Returns:
    matplotlib.figure.Figure: The plotted figure.
    '''
    urban_area_legend = False
    not_urban_area_legend = False
    is_rural = urban_vicinity['urmask'] == 0
    is_urban = urban_vicinity['urmask'] == 1

    if os.path.exists(cache):
        ds = xr.open_dataset(cache)
        rural_anomaly = ds['rural_anomaly']
        urban_anomaly = ds['urban_anomaly']
        rural_mean = ds['rural_mean']
        urban_mean = ds['urban_mean']
    else:
        rural_mean = (ds_var[variable]
            .where(is_rural)
            .groupby('time.month')
            .mean(dim = [ds_var.cf['Y'].name, ds_var.cf['X'].name, 'time'])
            .compute()
        )
        urban_mean = (ds_var[variable]
            .where(is_urban)
            .groupby('time.month')
            .mean(dim = [ds_var.cf['Y'].name, ds_var.cf['X'].name, 'time'])
            .compute()
        )                    
        ds_var_period_mean = ds_var.groupby('time.month').mean('time')                  
        ds_annomaly = ds_var_period_mean[variable] - rural_mean
        rural_anomaly = ds_annomaly.where(is_rural)
        urban_anomaly = ds_annomaly.where(is_urban)   
        if cache != '':
            xr.Dataset(dict(
                rural_anomaly = rural_anomaly,
                urban_anomaly = urban_anomaly,
                rural_mean = rural_mean,
                urban_mean = urban_mean
            )).to_netcdf(cache)
            
    # Plot mean annual cycle (urban and rural)
    fig, ax = plt.subplots(figsize=(15, 7)) 
    (urban_mean-rural_mean).plot(ax=ax,  color = 'r', linestyle='-', 
                                     linewidth = 4, label='Urban mean')
                         
    (rural_mean-rural_mean).plot(ax=ax,  color = 'b', linestyle='-', 
                                 linewidth = 4, label='Vicinity mean')
                         
    # Plot individual data squares for urban and rural areas if requested
    if data_squares:
        # Fill within percentiles
        axis = [rural_anomaly.get_axis_num(rural_anomaly.cf['X'].name),
                rural_anomaly.get_axis_num(rural_anomaly.cf['Y'].name)]
        colors = ['blue', 'red']
        for index, anom in enumerate([rural_anomaly, urban_anomaly]): 
            lower_percentile = np.nanpercentile(anom, percentile, axis=axis)
            upper_percentile = np.nanpercentile(anom, 100-percentile, axis=axis)
            ax.fill_between(
                rural_anomaly['month'],
                lower_percentile, upper_percentile,
                color=colors[index], alpha=0.1
            )
            for i, j in product(anom.cf['X'].values, anom.cf['Y'].values):
                anom_val = anom.sel({ds_var.cf['X'].name:i,
                                     ds_var.cf['Y'].name:j})
                if not np.isnan(anom_val[0]):
                    anom_val.plot(ax=ax, color=colors[index], linewidth=0.5)
                         
    #Plot the observation if requested
    if not isinstance(valid_stations, list):
        codes_ins_city = valid_stations.code[valid_stations['inside_city'] == True]
        codes_out_city = valid_stations.code[valid_stations['inside_city'] == False]
        time_series_mon = time_series.groupby(time_series.index.month).mean()
        time_series_mon_mean = pd.DataFrame(index = time_series_mon.index)
        time_series_mon_mean['rural_mean'] = time_series_mon[codes_out_city].mean(axis = 1).values
        time_series_mon_mean['urban_mean'] = time_series_mon[codes_ins_city].mean(axis = 1).values
        time_series_anomaly = time_series_mon.sub(time_series_mon_mean['rural_mean'], axis = 0)
        time_series_mon_mean_anom = time_series_mon_mean.sub(time_series_mon_mean['rural_mean'], axis = 0)
     
        time_series_anomaly[codes_ins_city].plot(ax = ax, marker='o', color = 'k', 
                                                 linestyle='--', linewidth = 2)
        time_series_anomaly[codes_out_city].plot(ax = ax, marker='o', color = 'g', 
                                                 linestyle='--', linewidth = 2)
    
        time_series_mon_mean_anom['urban_mean'].plot(ax = ax, color='k', linestyle='-', 
                                                     linewidth = 4, label='Urban obs. mean', 
                                                     zorder = 2000) 
        time_series_mon_mean_anom['rural_mean'].plot(ax = ax, color='g', linestyle='-', 
                                                     linewidth = 4, label='Vicinity obs. mean', 
                                                     zorder = 2000)
        if cache != '':
            xr.Dataset(dict(
                urban_mean_anom = time_series_mon_mean_anom['urban_mean'],
                )).to_netcdf(cache)
    # Add legend to the plot
    ax.legend(fontsize = 14, loc='center left', bbox_to_anchor=(0, -0.2), prop={'size': 14})
    ax.set_ylim(-0.0015, 0.)
    0.002,-0.0015
    
    # Customize the plot
    #ax.set_xlabel('Month', fontsize = 18)
    # Dynamically get the unit from the dataset
    unit = ds_var[variable].attrs.get('units', 'unknown')  # Default to 'unknown' if 'units' is missing
    
    if variable == 'tasmin':
        ax.set_title(f"Minimum temperature anomaly for {city}", fontsize=18)
        ax.set_ylabel(f"Minimum temperature anomaly ({unit})", fontsize=18)
    elif variable == 'tasmax':
        ax.set_title(f"Maximum temperature anomaly for {city}", fontsize=18)
        ax.set_ylabel(f"Maximum temperature anomaly ({unit})", fontsize=18)
    elif variable == 'huss':
        ax.set_title(f"Atmospheric moisture anomaly for {city}", fontsize=18)
        ax.set_ylabel(f"Atmospheric moisture anomaly ({unit})", fontsize=18)
    elif variable == 'hurs':
        ax.set_title(f"Relative atmospheric humidity anomaly for {city}", fontsize=18)
        ax.set_ylabel(f"Relative atmospheric humidity ({unit})", fontsize=18)
    elif variable == 'sfcWind':
        ax.set_title(f"Wind speed anomaly for {city}", fontsize=18)
        ax.set_ylabel(f"Wind speed anomaly ({unit})", fontsize=18)
    elif variable == 'pr':
        ax.set_title(f"Precipitation anomaly for {city}", fontsize=18)
        ax.set_ylabel(f"Precipitation anomaly ({unit})", fontsize=18)


    ax.set_xticks(np.arange(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 
                        'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize = 18)
    ax.tick_params(axis='y', labelsize=18)
    
    return fig
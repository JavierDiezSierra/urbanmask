import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import os
import pandas as pd
import geopandas as gpd
import xarray as xr
from icecream import ic
from itertools import product
from shapely.geometry import Point, Polygon
import xesmf as xe

import cf_xarray  # This will attach the cf accessor to xarray objects


var_map = {
    'tasmin': 'TMIN',
    'tasmax': 'TMAX'
}

def compute_climatology(ds, variable, urban_vicinity):
    # Check if urban_vicinity is a dataset
    if isinstance(urban_vicinity, xr.Dataset):
        # Compute the mean of the dataset over the 'time' dimension
        ds_var_period_mean = ds.mean('time').compute()
        
        # Calculate rural mean where 'urmask' is 0 (indicating rural areas)
        rural_mean = ds_var_period_mean[variable].where(
            urban_vicinity['urmask'] == 0).mean().compute()
        
        # Calculate the anomaly by subtracting rural mean from the variable mean
        ds_anomaly = ds_var_period_mean[variable] - rural_mean
        
        # Compute the maximum absolute value of the anomalies
        max_abs_value = abs(ds_anomaly).max().item()
        
    else:
        # Compute the mean of the dataset over the 'time' dimension
        ds_var_period_mean = ds.mean('time').compute()
        
        # Extract latitude and longitude coordinates from the dataset
        lon_vals = ds_var_period_mean['lon'].values
        lat_vals = ds_var_period_mean['lat'].values
        
        # Create a boolean mask indicating if points are within the non-urban area
        mask = np.zeros((len(lat_vals), len(lon_vals)), dtype=bool)  # Initialize mask as False
        
        # Iterate over the coordinates and check if they are within the non-urban polygon
        for i in range(len(lat_vals)):
            for j in range(len(lon_vals)):
                point = Point(lon_vals[j], lat_vals[i])  # Pair each latitude with its corresponding longitude
                if urban_vicinity.contains(point).all():  # Check if the point is inside the non-urban area
                    mask[i, j] = True   # Mark as True if the point is within the polygon
        
        # Filter the dataset using the mask to get rural values
        rural_values = ds_var_period_mean[variable].where(mask)
        
        # Calculate the mean of the non-null values (those within the non-urban area)
        rural_mean = rural_values.mean().compute()
    
        # Calculate the anomaly by subtracting rural mean from the variable mean
        ds_anomaly = ds_var_period_mean[variable] - rural_mean
    
    return ds_anomaly  # Return the computed anomalies
    
def compute_time_series(ds, urban_vicinity, variable, cache=''):
    """
    Compute the time series of urban and rural anomalies for a dataset.

    Parameters:
    ds: xarray dataset
        The dataset containing the climate data.
    urban_vicinity: xarray dataset or list
        Either the dataset with urban mask or a list with gdf_urban and gdf_non_urban polygons.
    variable: str
        The variable name in the dataset to compute the anomalies.
    cache: str
        Filepath to store the computed results for caching (optional).
        
    Returns:
    rural_anomaly: xarray DataArray
        The time series anomaly for rural areas.
    urban_anomaly: xarray DataArray
        The time series anomaly for urban areas.
    """
    # Check if urban_vicinity is a dataset or a list of polygons
    if isinstance(urban_vicinity, xr.Dataset):
        if 'x' in urban_vicinity.dims and 'y' in urban_vicinity.dims:
            # Create regridder to match urmask with ds_RCM's 1D lat/lon grid
            regridder = xe.Regridder(urban_vicinity, ds, method='nearest_s2d')
            
            # Regrid urmask to the grid of ds_RCM
            urban_vicinity= regridder(urban_vicinity)

        is_rural = urban_vicinity['urmask'] == 0
        is_urban = urban_vicinity['urmask'] == 1
    else:
        # If urban_vicinity is a list of gdf_urban and gdf_non_urban polygons
        gdf_urban, gdf_non_urban = urban_vicinity

        # Create masks based on whether points are within the urban or non-urban areas
        lon_vals = ds['lon'].values
        lat_vals = ds['lat'].values
        
        is_rural = xr.DataArray(
            [[gdf_non_urban.contains(Point(lon, lat)).all() for lon in lon_vals] for lat in lat_vals],
            dims=["lat", "lon"]
        )
        is_urban = xr.DataArray(
            [[gdf_urban.contains(Point(lon, lat)).all() for lon in lon_vals] for lat in lat_vals],
            dims=["lat", "lon"]
        )
    # Check if cached file exists
    if os.path.exists(cache):
        ds_cached = xr.open_dataset(cache)
        rural_anomaly = ds_cached['rural_anomaly']
        urban_anomaly = ds_cached['urban_anomaly']
        rural_mean = ds_cached['rural_mean']
        urban_mean = ds_cached['urban_mean']
    else:
        # Compute the mean for rural and urban areas grouped by month
        rural_mean = (ds[variable]
            .where(is_rural)
            .groupby('time.month')
            .mean(dim=['lat', 'lon', 'time'])
            .compute()
        )
        urban_mean = (ds[variable]
            .where(is_urban)
            .groupby('time.month')
            .mean(dim=['lat', 'lon', 'time'])
            .compute()
        )                    
        
        # Compute the monthly mean over the whole dataset
        ds_var_period_mean = ds.groupby('time.month').mean('time')

        # Compute the anomaly as the difference from rural mean
        ds_anomaly = ds_var_period_mean[variable] - rural_mean
        rural_anomaly = ds_anomaly.where(is_rural)
        urban_anomaly = ds_anomaly.where(is_urban) 

        if 'x' in rural_anomaly.dims and 'y' in rural_anomaly.dims:
            rural_anomaly = rural_anomaly.rename({'y': 'Y', 'x': 'X'})
            urban_anomaly = urban_anomaly.rename({'y': 'Y', 'x': 'X'})
            
        elif 'lat' in rural_anomaly.dims and 'lon' in rural_anomaly.dims:
            rural_anomaly = rural_anomaly.rename({'lat': 'Y', 'lon': 'X'})
            urban_anomaly = urban_anomaly.rename({'lat': 'Y', 'lon': 'X'})
            
        # Cache the results if cache is provided
        if cache != '':
            xr.Dataset(dict(
                rural_anomaly=rural_anomaly,
                urban_anomaly=urban_anomaly,
                rural_mean=rural_mean,
                urban_mean=urban_mean
            )).to_netcdf(cache)
    
    return rural_anomaly, urban_anomaly,rural_mean, urban_mean

def plot_climatology(ds_anomaly, urban_vicinity, variable, URBAN, ucdb_city = [], 
                     valid_stations = [], time_series = [], city = None,
                     alpha_urb_borders = 1, 
                     linewidth_urb_borders = 2):
    """
    Plot the climatological data.

    Parameters:
        ds_anomaly (xr.Dataset): Dataset containing the climatological data.
        ucdb_city (gpd.GeoDataFrame): GeoDataFrame of the city boundaries.
        urban_vicinity (object): Object representing urban vicinity.
        obs (pd.DataFrame, optional): DataFrame containing observational data (default is None).

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(12, 6))
                         
    # Compute the maximum absolute value
    max_abs_value = abs(ds_anomaly).max().item()
                         
    if not isinstance(valid_stations, list):
        # calculate climatology and anomaly from the series
        #codes_ins_city = valid_stations.code[valid_stations['inside_city'] == True]
        #codes_out_city = valid_stations.code[valid_stations['inside_city'] == False]
        #time_series_rural_mean = pd.DataFrame(index = time_series.index)
        #time_series_rural_mean['rural_mean'] = time_series[codes_out_city].mean(axis = 1).values
        #time_series_anomaly = time_series.sub(time_series_rural_mean['rural_mean'], axis = 0)

        #ax.scatter(valid_stations.lon, valid_stations.lat, 
        #           c = time_series_anomaly.mean(axis = 0, skipna=True), 
        #           marker='o', cmap='bwr', 
        #           s = 40, edgecolors = 'gray', 
        #           vmin = -max_abs_value, vmax = max_abs_value,
        #           zorder = 10000) 
        ax.scatter(valid_stations['LON'], valid_stations['LAT'], c='grey',
                   marker='o', s=20, transform=ccrs.PlateCarree(),zorder = 10000) 


    im1 = ax.pcolormesh(ds_anomaly.lon, ds_anomaly.lat, ds_anomaly.values,
                    cmap='bwr', alpha = 0.7,
                    vmin = - max_abs_value, 
                    vmax = max_abs_value)
    
    cbar = fig.colorbar(im1, ax = ax)
    cbar.set_label('°C', rotation = 90, fontsize = 14)
    
    if not isinstance(valid_stations, list):#change
        ucdb_city.plot(ax=ax, facecolor="none", transform=proj, edgecolor="Green", linewidth=2, zorder = 1000)
    
    ax.coastlines()
    if variable == 'tasmin':
        ax.set_title(f"Minimum temperature anomaly for {city}", fontsize = 14)
    elif variable == 'tasmax':
        ax.set_title(f"Maximum temperature anomaly for {city}", fontsize = 14)    
    
    # Overlay the cell borders and handle NaNs
    #URBAN.plot_urban_borders(urban_vicinity, ax, 
    #                         alpha_urb_borders, 
    #                         linewidth_urb_borders)
    URBAN.plot_urban_polygon(urban_vicinity, ax)
    
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    return fig

def plot_time_series(rural_anomaly, urban_anomaly, rural_mean, urban_mean, variable, 
                     time_series=[], valid_stations=[], 
                     data_squares=False, percentile=100, 
                     var_map=var_map, ucdb_city=None, 
                     city=None, cache=''):
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
            
    # Plot mean annual cycle (urban and rural)
    fig, ax = plt.subplots(figsize=(15, 7)) 
    (urban_mean - rural_mean).plot(ax=ax, color='r', linestyle='-', 
                                   linewidth=4, label='Urban mean')
                         
    (rural_mean - rural_mean).plot(ax=ax, color='b', linestyle='-', 
                                   linewidth=4, label='Vicinity mean')
                         

    if data_squares:
        # Fill within percentiles
        axis = [rural_anomaly.get_axis_num(rural_anomaly.cf['X'].name),
                rural_anomaly.get_axis_num(rural_anomaly.cf['Y'].name)]
                
                
        colors = ['blue', 'red']
        # Loop over rural_anomaly and urban_anomaly
        for index, anom in enumerate([rural_anomaly, urban_anomaly]):
            lower_percentile = np.nanpercentile(anom, percentile, axis=axis)
            upper_percentile = np.nanpercentile(anom, 100-percentile, axis=axis)
            ax.fill_between(
                rural_anomaly['month'],
                lower_percentile, upper_percentile,
                color=colors[index], alpha=0.1
            )
            # Iterate over all the (X, Y) points using cf-xarray
            for i, j in product(anom.cf['X'].values, anom.cf['Y'].values):
                # Select the anomaly value for the given (i, j) point using cf names
                anom_val = anom.sel({anom.cf['X'].name: i, anom.cf['Y'].name: j})
                
                # Check if the value is not NaN
                if not np.isnan(anom_val.values).any():
                    # Plot the anomaly value on the provided axis
                    anom_val.plot(ax=ax, color=colors[index], linewidth=0.5)

    # Plot the observation if requested

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
    
    # Customize the plot
    #ax.set_xlabel('Month', fontsize = 18)
    if variable == 'tasmin' or variable == 'tn':
        ax.set_title(f"Minimum temperature anomaly for {city}", fontsize = 18)
        ax.set_ylabel(f"Minimum temperature anomaly (°C)", fontsize = 18)
    elif variable == 'tasmax' or variable == 'tx':
        ax.set_title(f"Maximum temperature anomaly for {city}", fontsize = 18)
        ax.set_ylabel(f"Maximum temperature anomaly (°C)", fontsize = 18)
    
    ax.set_xticks(np.arange(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 
                        'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize = 18)
    ax.tick_params(axis='y', labelsize=18)
    
    return fig

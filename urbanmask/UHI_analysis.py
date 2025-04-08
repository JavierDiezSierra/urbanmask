import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import os
import pandas as pd
import xarray as xr
from icecream import ic
from itertools import product
from urbanmask.urban_areas import plot_urban_polygon

class UrbanIsland:
    def __init__(
        self,
        *,
        ds = None, 
        urban_vicinity = None, 
        anomaly = 'abs',
        obs_attributes = None, 
        obs_timeseries = None, 
        ):
        """
        """        
        self.ds = ds
        self.urban_vicinity = urban_vicinity
        self.anomaly = anomaly
        self.obs_attr = obs_attributes or []
        self.obs_time = obs_timeseries or []

    def compute_spatial_climatology(self):
        """
        """
        # calculate climatology and anomaly from the model
        ds_var_period_mean = self.ds.mean('time').compute()
        rural_mean = ds_var_period_mean.where(
            self.urban_vicinity['urmask'] == 0).mean().compute()
        ds_anomaly = ds_var_period_mean - rural_mean        
        ds_anomaly.attrs['units'] = self.ds.attrs.get('units', 'unknown')

        # calculate climatology from the observations
        obs_anomaly = None
        if self.obs_attr:
            obs_period_mean = self.obs_time.mean()
            obs_rur_mean = obs_period_mean[self.obs_attr["Rural"] == True].mean(axis = 1)
            obs_anomaly = obs_period_mean - obs_rur_mean

        if self.anomaly == 'rel':
            ds_anomaly = (ds_anomaly/ds_var_period_mean)*100
            ds_anomaly.attrs['units'] = "%"
            if obs_anomaly is not None:
                obs_anomaly = (obs_anomaly/obs_period_mean)*100

        self.ds_spatial_climatology = ds_anomaly.compute()
        if obs_anomaly is not None:
            self.obs_spatial_climatology = obs_anomaly.compute()
        
    def compute_annual_cycle(self):
        """
        """
        is_rural = self.urban_vicinity['urmask'] == 0
        is_urban = self.urban_vicinity['urmask'] == 1
        rural_mean = (self.ds
            .where(is_rural)
            .groupby('time.month')
            .mean(dim = [self.ds.cf['Y'].name, self.ds.cf['X'].name, 'time'])
            .compute()
        )       
        ds_period_mean = self.ds.groupby('time.month').mean('time')                  
        ds_anomaly = ds_period_mean - rural_mean
        ds_anomaly.attrs['units'] = self.ds.attrs.get('units', 'unknown')

        if self.anomaly == 'rel':
            ds_anomaly = (ds_anomaly / ds_period_mean) * 100
            ds_anomaly.attrs['units'] = "%"
              
        self.ds_annual_cycle = ds_anomaly
    
    def plot_UI_map(
        self,
        *,
        ds_anomaly = None,
        obs_anomaly = None,
        city_name = None,
        alpha_urb_borders = 1, 
        linewidth_urb_borders = 2, 
        vmax = None,
        ):
        """
        """
        
        if ds_anomaly is None:
            if not hasattr(self, 'ds_spatial_climatology'):
                self.compute_spatial_climatology()
            ds_anomaly = self.ds_spatial_climatology
            
        if self.obs_attr:
            if obs_anomaly is None:
                if not hasattr(self, 'obs_spatial_climatology'):
                    self.compute_spatial_climatology()
                obs_anomaly = self.obs_spatial_climatology
        
        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(12, 6))
    
        # Compute the maximum absolute value
        if vmax:
            max_abs_value = vmax
        else:
            max_abs_value = abs(ds_anomaly).max().item()
        
        im1 = ax.pcolormesh(ds_anomaly.lon, ds_anomaly.lat, 
                            ds_anomaly.values,
                            cmap='bwr', alpha = 0.7,
                            vmin = -max_abs_value, vmax = max_abs_value)

        if obs_anomaly is not None:
            
            sc = ax.scatter(
                self.obs_attr.lon, 
                self.obs_attr.lat,
                c=obs_anomaly.values, 
                cmap='bwr',
                vmin=-max_abs_value, vmax=max_abs_value,
                marker='o', 
                s=40, 
                edgecolors='gray', 
                zorder=10000
            )
        
        cbar = fig.colorbar(im1, ax = ax)
        unit = ds_anomaly.attrs.get('units', 'unknown') 
        
        cbar.set_label(f"{unit}", rotation = 90, fontsize = 14)
        ax.set_title(f"Urban Island for {city_name} (variable: {ds_anomaly.name})", fontsize=18)

        ax.coastlines()
        plot_urban_polygon(self.urban_vicinity, ax)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        
        return fig

    def plot_UI_annual_cycle(
        self,
        *,
        ds_anomaly = None,
        obs_anomaly = None,
        percentiles = [5],                      
        gridcell_series = True, 
        city_name = None, 
        ax = None,            
        vmax = None, 
        vmin = None):
        '''
        '''
        if ds_anomaly is None:
            if not hasattr(self, 'ds_annual_cycle'):
                self.compute_annual_cycle()
            ds_anomaly = self.ds_annual_cycle
        
        is_rural = self.urban_vicinity['urmask'] == 0
        is_urban = self.urban_vicinity['urmask'] == 1

        rural_anomaly = ds_anomaly.where(is_rural)
        urban_anomaly = ds_anomaly.where(is_urban)
    
        urban_mean = urban_anomaly.mean(dim = [ds_anomaly.cf['Y'].name, ds_anomaly.cf['X'].name]).compute()
        rural_mean = rural_anomaly.mean(dim = [ds_anomaly.cf['Y'].name, ds_anomaly.cf['X'].name]).compute()

        urban_area_legend = False
        not_urban_area_legend = False
    
        fig, ax = plt.subplots(figsize=(15, 7))

        (urban_mean).plot(ax=ax,  color = 'r', linestyle='-', 
                                         linewidth = 4, label='Urban mean')
                             
        (rural_mean-rural_mean).plot(ax=ax,  color = 'b', linestyle='-', 
                                     linewidth = 4, label='Vicinity mean')
                             
        # Plot individual data squares for urban and rural areas if requested
        if percentiles:
            # Fill within percentiles
            axis = [rural_anomaly.get_axis_num(rural_anomaly.cf['X'].name),
                    rural_anomaly.get_axis_num(rural_anomaly.cf['Y'].name)]
            colors = ['blue', 'red']
            for index, anom in enumerate([ rural_anomaly, urban_anomaly]):
                for percentile in percentiles:
                    lower_percentile = np.nanpercentile(anom, percentile, axis=axis)
                    upper_percentile = np.nanpercentile(anom, 100-percentile, axis=axis)
                    ax.fill_between(
                        rural_anomaly['month'],
                        lower_percentile, upper_percentile,
                        color=colors[index], alpha=0.1, linewidth=1, linestyle = '--',
                    )
        
                    # Plot the lower percentile line
                    ax.plot(
                        rural_anomaly['month'], lower_percentile,
                        color=colors[index], alpha=0.5, linewidth=1, linestyle='--', label=f'Lower Percentile')
                    # Plot the upper percentile line
                    ax.plot(
                        rural_anomaly['month'], upper_percentile,
                        color=colors[index], alpha=0.5, linewidth=1, linestyle='--', label=f'Upper Percentile')
                for i, j in product(anom.cf['X'].values, anom.cf['Y'].values):
                    anom_val = anom.sel({ds_anomaly.cf['X'].name:i,
                                         ds_anomaly.cf['Y'].name:j})
                    if not np.isnan(anom_val[0]):
                        anom_val.plot(ax=ax, color=colors[index], linewidth=0.1, alpha = 0.1)
                             
        #Plot the observation if requested
        if self.obs_attr:
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
                
            
        ax.set_xticks(np.arange(1, 13))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 
                            'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize = 18)
        ax.tick_params(axis='y', labelsize=18)
        if vmax is not None and vmin is not None:
            ax.set_ylim(vmin, vmax)

        # Add legend to the plot
        ax.legend(fontsize = 14, loc='center left', bbox_to_anchor=(0, -0.2), prop={'size': 14})

        ax.set_title(f"Urban Island for {city_name} (variable: {ds_anomaly.name})", fontsize=18)

        unit = ds_anomaly.attrs.get('units', 'unknown')         
        ax.set_ylabel(f"{unit}")
      
        return fig

def plot_daily_cycle(variable, ds_var=None, urban_vicinity=None, 
             time_series=[], valid_stations=[], 
             data_squares=True, percentiles=[], 
             city=None, ax=None, vmax=None, 
             vmin=None, period= 'Annual', anomaly = True):
    '''
    Plot daily cycle data with optional urban area overlay and additional time series overlay.

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
    
    if period == 'jja':
        ds_var = ds_var.sel(time=ds_var['time'].dt.month.isin([6,7,8]))
    if period == 'djf':
        ds_var = ds_var.sel(time=ds_var['time'].dt.month.isin([1,2,12]))
    is_rural = urban_vicinity['urmask'] == 0
    is_urban = urban_vicinity['urmask'] == 1
    rural_mean = (ds_var[variable]
        .where(is_rural)
        .groupby('time.hour')  # Group by daily hours
        .mean(dim=[ds_var.cf['Y'].name, ds_var.cf['X'].name, 'time'])
        .compute()
    )

    ds_var_period_mean = ds_var.groupby('time.hour').mean('time')
    # Group by daily hours
    if anomaly == True:
        ds_anomaly = ds_var_period_mean[variable] - rural_mean
        if variable in ['huss', 'pr', 'sfcWind']:
            ds_anomaly = (ds_anomaly / ds_var_period_mean[variable]) * 100
    else:
        ds_anomaly = ds_var_period_mean[variable]

    rural_anomaly = ds_anomaly.where(is_rural)
    urban_anomaly = ds_anomaly.where(is_urban)

    urban_mean = urban_anomaly.mean(dim=[ds_var.cf['Y'].name, ds_var.cf['X'].name]).compute()

    # Plot daily cycle (urban and rural)
    if ax is None: 
        fig, ax = plt.subplots(figsize=(15, 7))
    else:
        fig = None

    urban_mean.plot(ax=ax, color='r', linestyle='-', 
                    linewidth=4, label='Urban mean')
    if anomaly == True:
        (rural_mean - rural_mean).plot(ax=ax, color='b', linestyle='-', 
                                       linewidth=4, label='Vicinity mean')
    else:
        rural_mean.plot(ax=ax, color='b', linestyle='-', 
                                       linewidth=4, label='Vicinity mean')
    # Plot individual data squares for urban and rural areas if requested
    if data_squares:
        axis = [rural_anomaly.get_axis_num(rural_anomaly.cf['X'].name),
                rural_anomaly.get_axis_num(rural_anomaly.cf['Y'].name)]
        colors = ['blue', 'red']
        for index, anom in enumerate([rural_anomaly, urban_anomaly]):
            for percentile in percentiles:
                lower_percentile = np.nanpercentile(anom, percentile, axis=axis)
                upper_percentile = np.nanpercentile(anom, 100 - percentile, axis=axis)
                ax.fill_between(
                    rural_anomaly['hour'],
                    lower_percentile, upper_percentile,
                    color=colors[index], alpha=0.1, linewidth=1, linestyle='--',
                )
    
                ax.plot(
                    rural_anomaly['hour'], lower_percentile,
                    color=colors[index], alpha=0.7, linewidth=1, linestyle='--', label=f'Lower Percentile')
    
                ax.plot(
                    rural_anomaly['hour'], upper_percentile,
                    color=colors[index], alpha=0.7, linewidth=1, linestyle='--', label=f'Upper Percentile')
            for i, j in product(anom.cf['X'].values, anom.cf['Y'].values):
                anom_val = anom.sel({ds_var.cf['X'].name: i, ds_var.cf['Y'].name: j})
                if not np.isnan(anom_val[0]):
                    anom_val.plot(ax=ax, color=colors[index], linewidth=0.1, alpha=0.1)

    # Plot observations if requested
    if not isinstance(valid_stations, list):
        codes_city = valid_stations.code[valid_stations['city'] == city]
        time_series_hour = time_series.groupby(time_series.index.hour).mean()
        time_series_hour_mean = pd.DataFrame(index=time_series_hour.index)
        time_series_hour_mean['urban_mean'] = time_series_hour[codes_city].mean(axis=1).values
        time_series_anomaly = time_series_hour.sub(time_series_hour_mean['urban_mean'], axis=0)

        time_series_anomaly[codes_city].plot(ax=ax, marker='o', color='k', 
                                             linestyle='--', linewidth=2)

        time_series_hour_mean['urban_mean'].plot(ax=ax, color='k', linestyle='-', 
                                                 linewidth=4, label='Urban obs. mean', 
                                                 zorder=2000)

    ax.set_xticks(np.arange(0, 24))
    ax.set_xticklabels([f'{h}' for h in range(24)], fontsize=12)
    ax.tick_params(axis='y', labelsize=18)
    if vmax is not None and vmin is not None:
        ax.set_ylim(vmin, vmax)
        
    unit = ds_var.attrs.get('units', 'unknown')

    if fig:
        ax.legend(fontsize=14, loc='center left', bbox_to_anchor=(0, -0.2), prop={'size': 14})

        if variable == 'tas':
            ax.set_title(f"Mean temperature daily cycle for {city} ({period})", fontsize=18)
            ax.set_ylabel(f"Mean temperature anomaly ({unit})", fontsize=18)
        elif variable == 'huss':
            ax.set_title(f"Atmospheric moisture daily cycle ({period})", fontsize=18)
            ax.set_ylabel(f"Atmospheric moisture anomaly (% relative to {unit})", fontsize=18)
        elif variable == 'hurs':
            ax.set_title(f"Relative atmospheric humidity daily cycle ({period})", fontsize=18)
            ax.set_ylabel(f"Relative atmospheric humidity ({unit})", fontsize=18)
        elif variable == 'sfcWind':
            ax.set_title(f"Wind speed daily cycle for ({period})", fontsize=18)
            ax.set_ylabel(f"Wind speed anomaly (% relative to {unit})", fontsize=18)
        elif variable == 'pr':
            ax.set_title(f"Precipitation daily cycle for ({period})", fontsize=18)
            ax.set_ylabel(f"Precipitation anomaly (% relative to {unit})", fontsize=18)

        return fig
    else:
        ax.set_title(f"", fontsize=18)
        ax.set_ylabel(f"", fontsize=18)

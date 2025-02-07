import cartopy.crs as ccrs
import dask
import pandas as pd
import geopandas as gpd
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import xarray as xr
from icecream import ic
from itertools import product
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.morphology import dilation, square, remove_small_objects

from shapely.geometry import box
from shapely.geometry import Polygon
from shapely.ops import unary_union

class Urban_vicinity:
    def __init__(
        self, 
        urban_th : float = 0.1,
        urban_sur_th : float = 0.1,
        orog_diff : float = 100.0,
        sftlf_th : float = 70,
        scale : float = 2.0, 
        min_city_size : int = 0, 
        lon_city : float | None = None,
        lat_city : float | None = None,
        lon_lim : float = 1.0,
        lat_lim : float = 1.0,
        model : str | None = None,
        domain : str | None = None,
        urban_var : str | None = None,
    ):
        """
        Hyperparameters requered for urban/rural area selection
    
        Parameters
        ----------
        urban_th : float
            Urban fracción threshold. Cells with urban fraccion values above this threshold are considered urban cells
        urban_sur_th : float
Urban surrounding threshold. Cells with urban fraction values below this threshold might be considered rural surrounding cells.
        orog_diff : float
Altitude difference (m) respects the maximum and minimum elevation of the urban cells.
        sftlf_th : float 
            Minimum fraction of land required to include a cell in the analysis
        scale : float 
            Ratio between rural surrounding  and urban grid boxes
        min_city_size : int
            Remove urban nuclei smaller than the specified size.
        lon_city : float
            Longitude of the city cente        
        lat_city : float
            Latitude of the city cente
        lon_lim : float
            Longitude limit of the study area respect to the city center. Cells outside this area are excluded from the analysis.
        lat_lim : float
            Latitude limit of the study area respect to the city center.
            Cells outside this area are excluded from the analysis.
        model : str
            GCM/RCM model name
        domain : str
            Mode domain (if applicable)
        """        
        self.urban_th = urban_th
        self.urban_sur_th = urban_sur_th
        self.orog_diff = orog_diff
        self.sftlf_th = sftlf_th
        self.scale = scale
        self.min_city_size = min_city_size
        self.lon_city = lon_city
        self.lat_city = lat_city
        self.lon_lim = lon_lim
        self.lat_lim = lat_lim
        self.model = model
        self.domain = domain
        self.urban_var = urban_var

    def crop_area_city(
        self, 
        ds : xr.DataArray | None = None,
        res : int | None = None
    ) -> xr.DataArray:
        """
        Select area around a central city point.
    
        Parameters
        ----------
        ds : xarray.DataArray 
            xarray with longitud and latitud.
        res : xarray.DataArray
            Domain resolution (e.g. 11/22).
            
        Returns
        -------
        ds : xarray.DataArray
            Cropped xarray.
        """
        # number of cells around the city
        dlon = int(111*self.lon_lim/res)
        dlat = int(111*self.lat_lim/res)
        # select point close the city
        dist = (ds['lon']-self.lon_city)**2 + (ds['lat']-self.lat_city)**2
        [ilat], [ilon] = np.where(dist == np.min(dist))
        if ds.lon.ndim == 2:
        # crop area
            ds = ds.isel(**{
            ds.cf['Y'].name: slice(ilat-dlat,ilat+dlat),
            ds.cf['X'].name : slice(ilon-dlon,ilon+dlon)
            })   
        else:
         		# Crop the area for the city using the domain resolution
            # Define trimming limits
            lat_min = self.lat_city - self.lat_lim
            lat_max = self.lat_city + self.lat_lim
            lon_min = self.lon_city - self.lon_lim
            lon_max = self.lon_city + self.lon_lim
            
            # Crop the dataset
            ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        return ds

    def define_masks(
        self, 
        ds_sftuf : xr.DataArray | None = None, 
        ds_orog : xr.DataArray | None = None, 
        ds_sftlf: xr.DataArray | None = None,
    )-> xr.DataArray:
        """
        Define masks for urban fraction, orography and land-sea mask.
    
        Parameters
        ----------
        ds_sftuf : xarray.DataArray 
            Urban fraction (0-1)
        ds_orog : xarray.DataArray
            Orogrhapy (m)
        ds_sftlf : xarray.DataArray
            Land-sea percentaje (%)
            
        Returns
        -------
        sftuf_mask : xarray.DataArray 
            Binary mask indicating urban areas with 1 and 0 for the rest.
        sftuf_sur_mask : xarray.DataArray 
            Binary mask indicating surroundings of urban areas affected by the urban effect with 1 and 0 for the rest.
        orog_mask : xarray.DataArray
            Binary mask indicating of orography with .
        sftlf_mask : xarray.DataArray
            Binary mask indicating sea areas.
        """
        # sftuf
        sftuf_mask = ds_sftuf[self.urban_var] > self.urban_th
        # Remove small objects
        sftuf_mask_rem_small = remove_small_objects(sftuf_mask.values.astype(bool), 
                                                    min_size = self.min_city_size)
        sftuf_mask.data = sftuf_mask_rem_small
        deleted_small = ~sftuf_mask_rem_small*(ds_sftuf[self.urban_var] > self.urban_th)
        # Calculate surrounding mask and delete small objects from it
        sftuf_sur_mask_1 = ds_sftuf[self.urban_var] <= self.urban_th
        sftuf_sur_mask_2 = ds_sftuf[self.urban_var] > self.urban_sur_th
        sftuf_sur_mask_th = sftuf_sur_mask_1*sftuf_sur_mask_2
        sftuf_sur_mask = xr.where(deleted_small, True, sftuf_sur_mask_th)
        # orog
        urban_elev_max = ds_orog["orog"].where(sftuf_mask).max().item()
        urban_elev_min = ds_orog["orog"].where(sftuf_mask).min().item()
        orog_mask1 = ds_orog["orog"] < (self.orog_diff + urban_elev_max)
        orog_mask2 = ds_orog["orog"] > (urban_elev_min - self.orog_diff)
        orog_mask = orog_mask1 & orog_mask2
        
        #sftlf
        sftlf_mask = ds_sftlf["sftlf"] > self.sftlf_th   
        
        # Apply orog and sftlf thresholds to the urban_mask
        sftuf_mask = sftuf_mask*sftlf_mask
        sftuf_sur_mask = sftuf_sur_mask*sftlf_mask

        self.urban_elev_min = urban_elev_min 
        self.urban_elev_max = urban_elev_max 
    
        return sftuf_mask, sftuf_sur_mask, orog_mask, sftlf_mask


    def select_urban_vicinity(
        self,
        sftuf_mask : xr.DataArray | None = None,
        orog_mask : xr.DataArray | None = None,
        sftlf_mask : xr.DataArray | None = None,
        sftuf_sur_mask : xr.DataArray | None = None,
        scale: int | None = None
    ) -> xr.DataArray:
        """
        Funtion to select a number of non-urban cells based on surrounding urban areas using a dilation operation and excluding large water bodies, mountains and small urban nuclei.
    
        Parameters
        ----------
        sftuf_mask : xarray.DataArray 
            Binary mask indicating urban areas with 1 and 0 for the rest.
        orog_mask : xarray.DataArray
            Binary mask indicating of orography with .
        sftlf_mask : xarray.DataArray
            Binary mask indicating sea areas.
        sftuf_sur_mask : xarray.DataArray 
            Binary mask indicating surroundings of urban areas affected by the urban effect with 1 and 0 for the rest.
        scale : int 
            Urban-rural ratio of grid boxes.
    
        Returns
        -------
        xarray.DataArray 
            Mask of urban (1) surrounding non-urban cells (0) and the rest (NaN).
        """
        def delete_surrounding_intersect(dilated_data, sftuf_sur_mask):
            """
            Delete surroundings intersecting with dilated data
            """
            # Delete surroundings which intersect with dilated data
            dilated_data_surr = dilated_data * sftuf_sur_mask.astype(int)
            dilated_data_surr_opposite = xr.where(dilated_data_surr == 0, 1, 
                                                  xr.where(dilated_data_surr == 1, 0, dilated_data_surr))
            dilated_data = dilated_data * dilated_data_surr_opposite
            return dilated_data
        
        kernel1 = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]])        
        kernel2 = np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]])
        
        if scale is None:
            scale = self.scale
        
        data_array = xr.DataArray(sftuf_mask).astype(int)

        urban_cells = np.sum(sftuf_mask).values
        non_urban_cells = 0
        counter = 0
        while non_urban_cells <= urban_cells * scale:
            # Dilation (Try with kernel 1)
            dilated_data = xr.apply_ufunc(dilation, 
                                          data_array if counter == 0 else dilated_data, 
                                          kwargs={'footprint': kernel1})
            # Delete fixed variables
            dilated_data = (dilated_data * orog_mask * sftlf_mask).astype(int)
            
            if np.sum(dilated_data) - urban_cells == non_urban_cells:
                #Try with kernel2
                dilated_data = xr.apply_ufunc(dilation, 
                                              data_array if counter == 0 else dilated_data, 
                                              kwargs={'footprint': kernel2})
                # Delete fixed variables
                dilated_data = (dilated_data * orog_mask * sftlf_mask).astype(int)
                
                if np.sum(dilated_data) - urban_cells  == non_urban_cells:
                    print(f"Warning: No more non-urban cells can be found in iteration number {counter}")
                    break
                    
            # Number of surrounding cells intersecting dilated data
            dilated_data_surr_cells = np.sum(dilated_data * sftuf_sur_mask.astype(int))
            non_urban_cells = (np.sum(dilated_data) - urban_cells).values - dilated_data_surr_cells
            counter += 1

        # Delete surrounding intersectig with dilated data
        dilated_data = delete_surrounding_intersect(dilated_data, sftuf_sur_mask)  
        # Assing rural cells (1), vicinity (0) and the rest (nan) 
        non_urban_mask = xr.DataArray(dilated_data.where(~sftuf_mask).fillna(0))
        urban_area = sftuf_mask.astype(int).where(sftuf_mask.astype(int) == 1, np.nan)
        urban_area = urban_area.where(non_urban_mask.astype(int) == 0, 0)
        urban_area = urban_area.to_dataset(name='urmask')
        # Add attributes
        urban_area = Urban_vicinity.netcdf_attrs(self, urban_area)
        
        return urban_area

    def plot_urban_polygon(self, ds, ax):
        '''
        '''
        # Assume the mask is in the 'urmask' variable
        mask = ds['urmask']
        lon2d = mask.lon.values
        lat2d = mask.lat.values
        
        # Create lists to store polygons for urban areas (mask == 1) and non-urban areas (mask == 0)
        urban_polygons = []
        non_urban_polygons = []
        
        if lon2d.ndim == 1:
            dist_lat = abs(lat2d[1] - lat2d[0]) / 2
            dist_lon = abs(lon2d[1] - lon2d[0]) / 2
            for lon in range(len(lon2d)):
                for lat in range(len(lat2d)):
                    square = Polygon([
                        (round(lon2d[lon] - dist_lon, 3), round(lat2d[lat] - dist_lat, 3)),  # bottom-left corner
                        (round(lon2d[lon] + dist_lon, 3), round(lat2d[lat] - dist_lat, 3)),  # bottom-right corner
                        (round(lon2d[lon] + dist_lon, 3), round(lat2d[lat] + dist_lat, 3)),  # top-right corner
                        (round(lon2d[lon] - dist_lon, 3), round(lat2d[lat] + dist_lat, 3)),  # top-left corner
                    ])

                    # Add the polygon to the corresponding list
                    if mask[lat, lon] == 1:
                        urban_polygons.append(square)
                    elif mask[lat, lon] == 0:
                        non_urban_polygons.append(square)
        else:
            dist_lat = abs(lat2d[1, 0] - lat2d[0, 0])/2
            dist_lon = abs(lon2d[0, 1] - lon2d[0, 0])/2
    
            # Iterate through the mask and generate polygons for urban (1) and non-urban (0) cells
            for lat in range(mask.shape[0] - 1):  # Avoid the last index to prevent out-of-bounds errors
                for lon in range(mask.shape[1] - 1):
                    # Create a polygon using the 2D lat/lon coordinates of the cell corners
                    if pd.isnull(mask.lon[lat, lon]): # If cell contains nans continue
                        continue
                    square = Polygon([
                        (mask.lon[lat, lon] - dist_lon, mask.lat[lat, lon] - dist_lat),          # bottom-left corner
                        (mask.lon[lat, lon + 1] - dist_lon, mask.lat[lat, lon + 1] - dist_lat),  # bottom-right corner
                        (mask.lon[lat + 1, lon + 1] - dist_lon, mask.lat[lat + 1, lon + 1] - dist_lat),  # top-right corner
                        (mask.lon[lat + 1, lon] - dist_lon, mask.lat[lat + 1, lon] - dist_lat),  # top-left corner
                    ])
                    # Add the polygon to the corresponding list
                    if mask[lat, lon] == 1:
                        urban_polygons.append(square)
                    elif mask[lat, lon] == 0:
                        non_urban_polygons.append(square)
                        
        # Unite all adjacent polygons for urban (mask == 1) and non-urban (mask == 0)
        unified_urban_polygon = unary_union(urban_polygons)
        unified_non_urban_polygon = unary_union(non_urban_polygons)
        # Create GeoDataFrames for the urban and non-urban polygons
        # CRS 'EPSG:4326' specifies the WGS 84 coordinate system, which is widely used for global GPS coordinates (lat/lon)
        gdf_urban = gpd.GeoDataFrame(geometry=[unified_urban_polygon])
        gdf_non_urban = gpd.GeoDataFrame(geometry=[unified_non_urban_polygon])
        # Plot the boundary of the unified non-urban polygon (in blue)
        gdf_non_urban.boundary.plot(ax=ax,color='b', zorder=1, linewidth=2)
        # Plot the boundary of the unified urban polygon (in red) on top of the non-urban
        gdf_urban.boundary.plot(ax=ax,  color='red', zorder=100, linewidth=2)

        return(gdf_urban, gdf_non_urban)
    
    def plot_urban_borders(self, ds, ax, alpha = 1, linewidth = 2):
        """
        Plot the borders of urban areas on a map.
    
        Parameters:
        ds (xr.Dataset): The dataset containing longitude, latitude, and urban area data.
        ax (matplotlib.axes._subplots.AxesSubplot): The matplotlib axes on which to plot.
    
        """
        lon2d = ds.lon.values
        lat2d = ds.lat.values
        dist_lat = lat2d[1, 0] - lat2d[0, 0]
        dist_lon = lon2d[0, 1] - lon2d[0, 0]
        dist_latlon = lat2d[0 ,1] - lat2d[0, 0]
        dist_lonlat = lon2d[1, 0] - lon2d[0, 0]
        # Overlay the cell borders and handle NaNs
        for i in range(len(ds.lat)-1):
            for j in range(len(ds.lon)-1):
                lons = [lon2d[i, j], lon2d[i, j+1], lon2d[i+1, j+1], lon2d[i+1, j], lon2d[i, j]]
                lats = [lat2d[i, j], lat2d[i, j+1], lat2d[i+1, j+1], lat2d[i+1, j], lat2d[i, j]]

                lons = lons - abs(lon2d[i, j] - lon2d[i, j+1])/2              
                lats = lats - abs(lat2d[i, j] - lat2d[i+1, j])/2
                
                data_cell = ds['urmask'].values[i, j]
                
                if data_cell == 1:
                    ax.plot(lons, lats, color='grey', zorder = 100, linewidth = linewidth, alpha = alpha)
                elif data_cell == 0:
                    ax.plot(lons, lats, color='green', zorder = 1, linewidth = linewidth, alpha = alpha)

         # Plot the rightmost column
        for i in range(len(ds.lat) - 1):
            lons = [lon2d[i, -1], lon2d[i + 1, -1], lon2d[i + 1, -1] + dist_lon, lon2d[i, -1] + dist_lon, lon2d[i, -1]]
            lats = [lat2d[i, -1], lat2d[i + 1, -1], lat2d[i + 1, -1] + dist_latlon, lat2d[i, -1] + dist_latlon, lat2d[i, -1]]

            lons = lons - abs(lon2d[i, -1] - lon2d[i, -1])/2 - dist_lon/2            
            lats = lats - abs(lat2d[i, -1] - lat2d[i+1, -1])/2 
            
            data_cell = ds['urmask'].values[i, -1]
        
            if data_cell == 1:
                ax.plot(lons, lats, color='grey', zorder=100, linewidth = linewidth, alpha = alpha)
            elif data_cell == 0:
                ax.plot(lons, lats, color='green', zorder=1, linewidth = linewidth, alpha = alpha)
        
        # Plot the topmost row
        for j in range(len(ds.lon) - 1):
            lons = [lon2d[-1, j], lon2d[-1, j + 1], lon2d[-1, j + 1]  + dist_lonlat, lon2d[-1, j]  + dist_lonlat, lon2d[-1, j]]
            lats = [lat2d[-1, j], lat2d[-1, j + 1], lat2d[-1, j + 1] + dist_lat, lat2d[-1, j] + dist_lat, lat2d[-1, j]]

            
            lons = lons - abs(lon2d[-1, j] - lon2d[-1, j+1])/2           
            lats = lats - abs(lat2d[-1, j] - lat2d[-1, j])/2 - dist_lat/2  
            
            data_cell = ds['urmask'].values[-1, j]
        
            if data_cell == 1:
                ax.plot(lons, lats, color='red', zorder=100, linewidth=2)
            elif data_cell == 0:
                ax.plot(lons, lats, color='b', zorder=1, linewidth=2)
        
        # Plot the bottom right corner
        lons = [
            lon2d[-1, -1],
            lon2d[-1, -1] + dist_lon,
            lon2d[-1, -1] + dist_lon  + dist_lonlat,
            lon2d[-1, -1]  + dist_lonlat,
            lon2d[-1, -1]
        ]
        lats = [
            lat2d[-1, -1],
            lat2d[-1, -1] + dist_latlon,
            lat2d[-1, -1] + dist_lat + dist_latlon,
            lat2d[-1, -1] + dist_lat,
            lat2d[-1, -1]
        ]
            
        data_cell = ds['urmask'].values[-1, -1]

        lons = lons - abs(lon2d[-1, -1] - lon2d[-1, -1])/2 - dist_lon/2
        lats = lats - abs(lat2d[-1, -1] - lat2d[-1, -1])/2 - dist_lat/2  
        
        if data_cell == 1:
            ax.plot(lons, lats, color='red', zorder=100, linewidth=2)
        elif data_cell == 0:
            ax.plot(lons, lats, color='b', zorder=1, linewidth=2)
            
    def plot_static_variables(self, ds_sftuf, ds_orog, ds_sftlf,
                             sftuf_mask, orog_mask, sftlf_mask, urban_areas = None,
                            ):
        """
        Plot fix variables including urban fraction, orography, and land-sea mask.
    
        Parameters:
        ds_sftuf (xr.Dataset): Dataset containing urban fraction data.
        ds_orog (xr.Dataset): Dataset containing orography data.
        ds_sftlf (xr.Dataset): Dataset containing land-sea mask data.
        sftuf_mask (xr.DataArray): Mask for urban fraction.
        orog_mask (xr.DataArray): Mask for orography.
        sftlf_mask (xr.DataArray): Mask for land-sea mask.
        urban_areas (xr.Dataset, optional): Dataset containing urban area borders.
                                            Defaults to None.
    
        Returns:
        matplotlib.figure.Figure: The generated figure.
        """
        colors = ['#7C5B49', '#92716B', '#A89080', '#C0B49E', '#DACCB9', '#F5F5DC']
        colors = ['#278908', '#faf998', '#66473b']
        custom_cmap = LinearSegmentedColormap.from_list("custom_terrain", colors)
        
        proj = ccrs.PlateCarree()
        fig, axes = plt.subplots(2, 3, subplot_kw={'projection': proj}, figsize=(20, 10))
                        
        im1 = axes[0, 0].pcolormesh(ds_sftuf.lon, ds_sftuf.lat,
                                    ds_sftuf[self.urban_var].values,
                                    cmap='binary',
                                    vmin = np.nanmin(ds_sftuf[self.urban_var]), 
                                    vmax = np.nanmax(ds_sftuf[self.urban_var]))
        cim1 = fig.colorbar(im1, ax=axes[0, 0], orientation='vertical')
        cim1.set_label("Units: %")
        axes[0, 0].set_title('Urban Fraction', fontsize = 14)
        axes[0, 0].coastlines()
        
        im2 = axes[0, 1].pcolormesh(ds_orog.lon, ds_orog.lat,
                                    ds_orog['orog'], 
                                    cmap=custom_cmap, 
                                    vmin = np.nanmin(ds_orog['orog']), 
                                    vmax = np.nanmax(ds_orog['orog'])
                                   )
        
        cim2 = fig.colorbar(im2, ax=axes[0, 1], orientation='vertical')
        cim2.set_label("Units: m")
        axes[0, 1].set_title('Orography', fontsize = 14)
        axes[0, 1].coastlines()
        
        im3 = axes[0, 2].pcolormesh(ds_sftlf.lon, ds_sftlf.lat,
                                    ds_sftlf["sftlf"],
                                    cmap='winter', 
                                    vmin = np.nanmin(ds_sftlf["sftlf"]), 
                                    vmax = np.nanmax(ds_sftlf["sftlf"])
                                   )
        cim3 = fig.colorbar(im3, ax=axes[0, 2], orientation='vertical')
        cim3.set_label("Units: %")
        axes[0, 2].set_title('Land-sea', fontsize = 14)
        axes[0, 2].coastlines()
    
        # masks
        im1 = axes[1, 0].pcolormesh(ds_sftuf.lon, ds_sftuf.lat,
                                    ds_sftuf[self.urban_var].where(sftuf_mask == 1, np.nan),
                                    cmap='binary',
                                    vmin = np.nanmin(ds_sftuf[self.urban_var]),
                                    vmax = np.nanmax(ds_sftuf[self.urban_var])
                                   )
        fig.colorbar(im1, ax=axes[1, 0], orientation='vertical')
        if not urban_areas:
            axes[1, 0].set_title('Urban Fraction\n(urb_th >' +  str(self.urban_th) + ')')
        else:
            axes[1, 0].set_title(f"Urban Fraction\n(urb_th = {self.urban_th}, urb_sur_th = {self.urban_sur_th}\nscale = {self.scale}, max_city = {self.min_city_size})", fontsize = 14)
        axes[1, 0].coastlines()

        im2 = axes[1, 1].pcolormesh(ds_orog.lon, ds_orog.lat,
                                    ds_orog['orog'].where(orog_mask == 1, np.nan), 
                                    cmap=custom_cmap, 
                                    vmin = np.nanmin(ds_orog['orog']), 
                                    vmax = np.nanmax(ds_orog['orog'])
                                   )
        fig.colorbar(im2, ax=axes[1, 1], orientation='vertical')
        elev_lim_min = self.urban_elev_min - self.orog_diff
        elev_lim_max = self.urban_elev_max + self.orog_diff
        axes[1, 1].set_title(f'Orography\n(orog_diff = {self.orog_diff};\n{elev_lim_min:.0f} m < orog < {elev_lim_max:.0f} m)', fontsize = 14)
        axes[1, 1].coastlines()

        im3 = axes[1, 2].pcolormesh(ds_sftlf.lon, ds_sftlf.lat,
                                    ds_sftlf["sftlf"].where(sftlf_mask == 1, np.nan),
                                    cmap='winter', 
                                    vmin = np.nanmin(ds_sftlf["sftlf"]), 
                                    vmax = np.nanmax(ds_sftlf["sftlf"])
                                   )
        fig.colorbar(im3, ax=axes[1, 2], orientation='vertical')
        axes[1, 2].set_title('Land-sea\n(sftlf_th= 70;\nsftlf >' + str(self.sftlf_th) + '%)', fontsize = 14)
        axes[1, 2].coastlines()

        if urban_areas:
            for k in range(3):
                #Urban_vicinity.plot_urban_borders(self, urban_areas, axes[1, k])
                Urban_vicinity.plot_urban_polygon(self, urban_areas, axes[1, k])


        plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust vertical and horizontal space
        return fig

    def netcdf_attrs(self, ds):        
        """
        Add metadata to urban area file.
    
        Parameters
        ----------
        urban_area : xarray.Dataset 
            Binary mask indicating urban areas (1), non-urban (vicinity) areas (0) and NaN for the rest.
        """
        # add attribtes
        ds['urmask'].attrs['long_name'] = 'Urban vs. vicinity. 1 corresponds to urban areas and 0 to the surrounding areas'
        
        attrs_list = ["urban_th", "urban_sur_th", "orog_diff", "sftlf_th", "sftlf_th", "scale", 
                      "min_city_size", "lon_city", "lat_city", "lon_lim", "lat_lim", "model", "domain"]
        
        for attr in attrs_list:
            if getattr(self, attr):
                ds['urmask'].attrs[attr] = getattr(self, attr)
            
        return ds


    def create_urban_dataset(self,ucdb_city, ds):
        '''
        Creates a dataset for urban areas where each cell value represents the percentage of the cell area
        that is within the city limits.
    
        Parameters:ssh-keygen -f "/home/yaizaquintana/.ssh/known_hosts" -R "ui.sci.unican.es"
        ucdb_city: GeoDataFrame containing the city's geometry.
        ds: Dataset from which we are extracting the coordinates.
    
        Returns:
            ds_urban: An xarray dataset with cells representing the percentage of urban area coverage.
        '''
    
        # Create a grid of latitude and longitude edges to define cell boundaries
        lon_grid, lat_grid = np.meshgrid(ds['lon'].values, ds['lat'].values)
    
        # Combine all geometries into a single geometry (in case there are multiple city polygons)
        city_geometry = ucdb_city.geometry.unary_union
    
        # Create an array to store the percentage of each cell covered by the city
        urban_data = np.zeros((ds['lat'].size, ds['lon'].size))
    
        # Calculate percentage for each cell
        for i in range(ds['lat'].size):
            for j in range(ds['lon'].size):
                # Define the boundaries of the cell as a polygon
                cell_poly = box(
                    lon_grid[i, j] - 0.5 * np.diff(ds['lon'])[0],  # left
                    lat_grid[i, j] - 0.5 * np.diff(ds['lat'])[0],  # bottom
                    lon_grid[i, j] + 0.5 * np.diff(ds['lon'])[0],  # right
                    lat_grid[i, j] + 0.5 * np.diff(ds['lat'])[0]   # top
                )
    
                # Calculate the area of intersection with the city geometry
                intersection_area = cell_poly.intersection(city_geometry).area
                cell_area = cell_poly.area
    
                # Calculate the percentage of the cell covered by the city
                urban_data[i, j] = (intersection_area / cell_area) * 100
    
        # Create the final xarray dataset containing the urban percentage information
        ds_urban = xr.Dataset(
            {
                'sftuf': (['lat', 'lon'], urban_data)
            },
            coords={
                'lat': ds['lat'],
                'lon': ds['lon'],
            }
        )
        
        return ds_urban
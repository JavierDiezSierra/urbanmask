import xarray as xr
import geopandas as gpd
import os

def traverseDir(root):
    for (dirpath, dirnames, filenames) in os.walk(root):
        for file in filenames:
            if file.endswith(('.nc')):
                yield os.path.join(dirpath, file)

def fix_360_longitudes(
    dataset: xr.Dataset, lonname: str = "lon"
) -> xr.Dataset:
    """
    Fix longitude values.

    Function to transform datasets where longitudes are in (0, 360) to (-180, 180).

    Parameters
    ----------
    dataset (xarray.Dataset): data stored by dimensions
    lonname (str): name of the longitude dimension

    Returns
    -------
    dataset (xarray.Dataset): data with the new longitudes
    """
    lon = dataset[lonname]
    if lon.max().values > 180 and lon.min().values >= 0:
        dataset[lonname] = dataset[lonname].where(lon <= 180, other=lon - 360)
    return dataset

def kelvin2degC(ds, variable):
    """
    Convert a variable's units from Kelvin to Celsius in an xarray Dataset.

    Parameters:
    ds (xr.Dataset): The dataset containing the variable to convert.
    variable (str): The name of the variable to convert.

    Returns:
    xr.Dataset: The dataset with the variable converted to Celsius.
    """
    if ds[variable].attrs.get('units') == 'K':
        ds[variable] = ds[variable] -273.15
        ds[variable].attrs['units'] = 'degC'
        
    return ds


def fix_360_longitudes(
    dataset: xr.Dataset, lonname: str = "lon"
) -> xr.Dataset:
    """
    Fix longitude values.

    Function to transform datasets where longitudes are in (0, 360) to (-180, 180).

    Parameters
    ----------
    dataset (xarray.Dataset): data stored by dimensions
    lonname (str): name of the longitude dimension

    Returns
    -------
    dataset (xarray.Dataset): data with the new longitudes
    """
    lon = dataset[lonname]
    if lon.max().values > 180 and lon.min().values >= 0:
        dataset[lonname] = dataset[lonname].where(lon <= 180, other=lon - 360)
    return dataset
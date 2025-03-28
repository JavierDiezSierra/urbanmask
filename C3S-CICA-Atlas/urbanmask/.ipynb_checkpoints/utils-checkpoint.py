import xarray as xr
import geopandas as gpd
import os
import yaml

class YAMLconfig(dict):
    def __init__(self, data):
        if isinstance(data, str):
            with open(data, 'r') as file:
                data = yaml.safe_load(file)
        super().__init__(data)
        
    def __getattr__(self, name):
        if name in self:
            value = self[name]
            if isinstance(value, dict):
                return YAMLconfig(value)
            return value
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        self[name] = value
        
    def __delattr__(self, name):
        del self[name]


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
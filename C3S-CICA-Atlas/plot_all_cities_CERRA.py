#!/usr/bin/env python
# coding: utf-8

import glob
import os
import papermill as pm
import sys

from icecream import ic
from utils import RCM_DICT
from utils import YAMLconfig

# Notebook paths
input_notebook = 'urban_area_selection_CERRA.ipynb'
output_notebook = 'urban_area_selection_CERRA_papermill.ipynb'

# Climate variable and expected output
variable = 'tasmin'
expected_figure_number = 7
model= "CERRA"

# Load cities configuration from YAML
cities = YAMLconfig('selected_cities_CERRA.yaml')

# Default thresholds and limits
default_urban_th = cities['DEFAULT']['urban_th']
default_urban_sur_th = cities['DEFAULT']['urban_sur_th']
default_orog_diff = cities['DEFAULT']['orog_diff']
default_sftlf_th= cities['DEFAULT']['sftlf_th']
default_lon_lim = cities['DEFAULT']['lon_lim']
default_lat_lim = cities['DEFAULT']['lat_lim']
default_min_city_size = cities['DEFAULT']['min_city_size']

# Iterate over cities and process data
for city in cities:
    if city == "DEFAULT":
        continue
    abbr_city = city

    # Generate parameters for the current city
    parameters = {
        'name' :  cities[city]['name'],
        'city': city.split('_')[0],
        'lon_city': cities[city]['lon'],
        'lat_city': cities[city]['lat'],
        'variable': variable,
        'urban_th': cities[city].get('urban_th', default_urban_th),
        'urban_sur_th': cities[city].get('urban_sur_th', default_urban_sur_th),
        'orog_diff': cities[city].get('orog_diff', default_orog_diff),
        'sftlf_th': cities[city].get('sftlf_th', default_sftlf_th),
        'lon_lim': cities[city].get('lon_lim', default_lon_lim),
        'lat_lim': cities[city].get('lat_lim', default_lat_lim),
        'min_city_size': cities[city].get('min_city_size', default_min_city_size),
    }



    # Update directory to include urban variable
    directory = f"results_CERRA/{abbr_city}"
    if len(glob.glob(f"{directory}/*.pdf")) == expected_figure_number:
        continue

    # Execute notebook using Papermill
    try:
        print(f'executing: {city}')
        pm.execute_notebook(
            input_path=input_notebook,
            output_path=output_notebook,
            parameters=parameters,
            kernel_name='python3'
        )
    except Exception as e:
        # Handle errors by saving a failed version of the output notebook
        output_notebook_failed = output_notebook.replace('.ipynb', f'_ERROR_{abbr_city}_CERRA.ipynb')
        os.system(f'cp {output_notebook} {output_notebook_failed}')
        print(f'Error executing notebook: {e}')

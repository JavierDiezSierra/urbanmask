import papermill as pm
import os

# Define the directory to search
root = "/lustre/gmeteo/WORK/DATA/C3S-CDS/C3S-CICA-Atlas/v2/"
domain = "CORDEX-EUR-11"  # "CORDEX-CORE"/"CORDEX-EUR-11"
scenarios = ["historical", "rcp26", "rcp45", "rcp85"]
variables = [
    "t",
    "tn",
    "tx",
    "tnn",
    "txx",
    "tx35",
    "tx40",
    "tr",
    "dtr",
    "fd",
    "hd",
    "cd",
    "huss",
    "sfcwind",
    "rsds",
    "psl",
]

# Base notebook name
notebook_input = "mask_CORDEX.ipynb"

# Execute Papermill for each file
for scenario in scenarios:
    for var in variables:
        if scenario == "historical":
            name = f"{var}_{domain}_historical_mon_197001-200512_v02.nc"
            if var in ['cd','hd']:
                name = f'{var}_{domain}_historical_yr_1970-2005_v02.nc'
            elif 'var'== 'sfcwind':
                name = f'sfcwind_{domain}_historical_mon_197001-200512_v02.nc'
        else:
            name = f"{var}_{domain}_{scenario}_mon_200601-210012_v02.nc"
            if var in ['cd','hd']:
                name = f'{var}_{domain}_{scenario}_yr_2006-2100_v02.nc'
            elif 'var'== 'sfcwind':
                name = f'sfcwind_{domain}_{scenario}_mon_200601-210012_v02.nc'
        
        output_notebook = f"mask_CORDEX_{name.replace('.nc', '')}.ipynb"
        print(f"Running {name} -> {output_notebook}")

        try:
            pm.execute_notebook(
                notebook_input,
                output_notebook,
                parameters={
                    "root": root,
                    "domain": domain,
                    "name": name,
                    "scenario": scenario,
                },
            )
        except Exception as e:
            print(f"Error executing {name}: {e}")
            continue  # Skip to the next iteration if there's an error

import papermill as pm
import os

root = "/lustre/gmeteo/WORK/DATA/C3S-CDS/C3S-CICA-Atlas/v2/"
domain = "CERRA"
variables = [
    "t", "tn", "tx", "tnn", "txx", "tx35", "tx40", "tr", "dtr", "fd",
    "hd", "cd", "huss", "sfcWind", "rsds", "psl",
]

# Base notebook name
notebook_input = "mask_CERRA.ipynb"

# Execute Papermill for each file
for var in variables:
    name = f"{var}_CERRA_mon_198501-202112_v02.nc"
    if var in ['cd','hd']:
        name = f'{var}_CERRA_yr_1985-2021_v02.nc'
    if var == 'sfcWind':
        name= "sfcwind_CERRA_mon_198501-202112_v02.nc"
    output_notebook = f"mask_CERRA_{name.rsplit('.', 1)[0]}.ipynb"  # Más seguro que replace('.nc', '')

    print(f"Running {name} -> {output_notebook}")

    try:
        pm.execute_notebook(
            notebook_input,
            output_notebook,
            parameters={
                "root": root,
                "domain": domain,
                "name": name,
            },
        )
    except Exception as e:
        print(f"Error executing {name}: {e}")
        continue  # Skip to the next iteration if there's an error

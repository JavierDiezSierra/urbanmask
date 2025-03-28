import nbformat
import subprocess

notebook_path = "interpoletion.ipynb"

# Opción 1: Usando nbconvert (sin parámetros, ejecuta en orden)
subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", notebook_path, "--output", "executed_notebook.ipynb"], check=True)

# Opción 2: Usando papermill si necesitas parámetros
# import papermill as pm
# pm.execute_notebook(
#     notebook_path,
#     "executed_notebook.ipynb",
#     parameters={"param_name": "value"}  # Reemplaza con los parámetros necesarios
# )

print("Notebook ejecutado correctamente.")

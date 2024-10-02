# Urban and rural surrounding areas selection


This repository contains functions to select urban a rural surrounding areas from Global or Regional Climate Models given a specific city.

It includes functionalities to assess the Urban Heat Iland (UHI).

## Repository Overview
### Key Components
1. **Morphological Dilation Function**: This function delimited urban and rural surroundinf cells from three static variables(sftuf, orog, sftlf).

    - sftuf: Fraction of urban land use.
    - orog: Orography (elevation).
    - sftlf: Land-sea mask.

2. **Analyzing UHI Effect**: The UHI effect can be assess 

    - tasmin: Minimum temperature.
    - tasmax: Maximum temperature.

3. **urban_area_selection.ipynb Notebook**: This Jupyter Notebook allows users to explore urban vs. rural areas for different cities and analyze the UHI effect in those areas.

## Algorithm description

The algorithm uses three static variables (urban fraction, orography and land-sea fraction) along with several hyperparameters (see Table 1) to determine which cells are considered urban and which their rural surroundings. First, the location of the city of interest (**lon_city** and **lat_city**) and the study area (**lon_lim** and **lat_lim**) must be defined in geographic coordinates. Only those cells inside the study area can be selected as urban or rural. Then, the urban fraction threshold (**urban_th**) defines the grid boxes that represent urban areas in the model. Cells with urban fraction values higher than the urban threshold are considered urban cells. Small satellite urban areas not connected to the city’s core can be excluded using the **min_city_size** parameter. This parameter eliminates small urban clusters, classifying them as neither urban nor rural. The parameter **urban_sur_th** creates a buffer zone around urban cells. Cells with urban fraction values between **urban_th** and **urban_sur_th**, which might be affected by the UHI effect and should not be considered rural surrounding areas, are excluded from the analysis. This parameter is particularly relevant for high-resolution climate models and only affects the results if the hyperparameter **urban_th**  and urban_sur_th  are different.

The algorithm identifies urban areas based on the **urban_th** parameter and uses the static variables orography and land-sea fraction to create three masks which serves to define which cells might be considered rural. The parameter **orog_diff** serves to exclude mountainous areas around the city by masking surrounding grid boxes with an altitude difference between the maximum and minimum elevation of the urban cells. Large water bodies, such as lakes, oceans and rivers, can be excluded using the parameter **sftlf_th** (note that these parameters also affect the urban areas). The grid boxes that comply with all the aforementioned criteria can be selected as rural surroundings in an iterative process based on a morphological dilation function from the urban cells. The ratio of the number of urban vs rural cells is defined by the parameter **scale**.

The morphological dilation function used to determine rural surrounding areas is performed using the scikit-image Python package (https://scikit-image.org/). This function sets the value of a pixel to the maximum over all pixel values within a local neighborhood centered around it. The values where the footprint is 1 define this neighborhood. Two shapes of footprints or kernels are implemented. For each iteration, a morphological dilation with a cross-shaped footprint is first applied, allowing 4-connected cells to be neighbors of any cell that touches one of their edges. If the cross-shaped footprint iteration does not select any rural cells, a square footprint is then applied, including 4 additional connected samples (edges and diagonals). The cross-shaped footprint might not return any rural surrounding cell for coarse resolution models. In each iteration, masked values including water bodies, elevation differences, and urban cells are excluded. The iterative process is finalized when the number of rural surrounding cells reaches the proportion defined by the scale parameter.

| **Hyperparameter** | **Description** |
|--------------------|-----------------|
| `lon_city` and `lat_city` | Longitude and latitude of the city center |
| `lon_lim` and `lat_lim` | Limits of the study area with respect to the city center (`lon_city` and `lat_city`). Cells outside this area are excluded from the analysis. |
| `urban_th` | Urban fraction threshold. Cells with urban fraction values above this threshold are considered urban cells. |
| `urban_sur_th` | Urban surrounding threshold. Cells with urban fraction values below this threshold might be considered rural surrounding cells. |
| `orog_diff` | Altitude difference (m) with respect to the maximum and minimum elevation of the urban cells. |
| `sftlf_th` | Minimum fraction of land required to include a cell in the analysis. |
| `min_city_size` | Remove urban nuclei smaller than the specified size. |
| `scale` | Ratio between rural surrounding and urban grid boxes. |






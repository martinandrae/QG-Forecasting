import xarray as xr
import numpy as np

# Path to the directory containing the NetCDF files
file_directory = "/proj/berzelius-2022-164/weather/weatherbench/"

var_name = "temperature_850"
extension = "hPa"

# Pattern to match all the NetCDF files in the directory
file_pattern = f"{file_directory}/{var_name}/{var_name}{extension}_*.nc"

# Combine the datasets along the 'time' dimension
combined_ds = xr.open_mfdataset(file_pattern, combine='by_coords')

# Convert the combined dataset to a NumPy array
# If you want to save a specific variable, like 'geopotential', select it
# For example: data_array = combined_ds['geopotential'].values
data_array = combined_ds.to_array().values

save_directory = '/proj/berzelius-2022-164/users/sm_maran/data/wb'
# Save the NumPy array to an .npy file
np.save(f'{save_directory}/{var_name}{extension}_1979-2018_5.625deg.npy', data_array)

print("Combined data saved as .npy successfully.")

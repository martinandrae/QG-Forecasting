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

# Something to fix the shape
# data_array = data_array.reshape(data_array.shape[1], -1)

save_directory = '/proj/berzelius-2022-164/users/sm_maran/data/wb'
# Save the NumPy array to an .npy file
np.save(f'{save_directory}/{var_name}{extension}_1979-2018_5.625deg.npy', data_array)

print("Combined data saved as .npy successfully.")


"""
import numpy as np
subd = '/proj/berzelius-2022-164/users/sm_maran/data/wb'

# Load the arrays from the .npy files
geopotential_500hPa = np.load(f'{subd}/geopotential_500hPa_1979-2018_5.625deg.npy')
print('loaded Geop')
temperature_850hPa = np.load(f'{subd}/temperature_850hPa_1979-2018_5.625deg.npy')
print('loaded temp')

# Ensure the arrays have compatible dimensions for concatenation
# For instance, assuming both arrays have the shape (time, lat, lon),
# they can be concatenated along a new axis (e.g., axis 0 or 1).
# Here, we'll concatenate them along a new axis 0:
combined_array = np.stack((geopotential_500hPa, temperature_850hPa), axis=1)

# Now the combined_array will have a shape of (2, time, lat, lon)
# Save the combined array to a new .npy file
np.save('combined_geopotential_temperature.npy', combined_array)

print("Combined array shape:", combined_array.shape)


# shp = combined_array.shape
# combined_array = combined_array.reshape(shp[0], shp[1],32,64)
"""
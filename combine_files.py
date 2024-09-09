import xarray as xr
import numpy as np



# Path to the directory containing the NetCDF files
file_directory = "/proj/berzelius-2022-164/weather/era5/weatherbench1/full_netcdf/"

var_name = "2m_temperature"
#field_name = "lsm" # For the constants data

extension = "" # Use "hPa" when needed

# Pattern to match all the NetCDF files in the directory
file_pattern = f"{file_directory}/{var_name}/{var_name}{extension}_*.nc"

# Combine the datasets along the 'time' dimension
combined_ds = xr.open_mfdataset(file_pattern, combine='by_coords')


# Convert the combined dataset to a NumPy array
# If you want to save a specific variable, like 'geopotential', select it
# For example: data_array = combined_ds['lsm'].values
#data_array = combined_ds[field_name].values
data_array = combined_ds.to_array().values

save_directory = '/proj/berzelius-2022-164/users/sm_maran/data/wb'
# Save the NumPy array to an .npy file
np.save(f'{save_directory}/{var_name}{extension}_1979-2018_5.625deg.npy', data_array)
#np.save(f'{save_directory}/{field_name}{extension}_1979-2018_5.625deg.npy', data_array)

print("Combined data saved as .npy successfully.")


"""

import numpy as np
import json

subd = '/proj/berzelius-2022-164/users/sm_maran/data/wb'

# Dictionary mapping file prefixes to their respective variable names for saving
var_names = {
    "geopotential_500": "z500",
    "temperature_850": "t850",
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    # "orography": "orog",
    # "lsm": "lsm",
}

# Initialize an empty list to store the loaded arrays
arrays = []
# Initialize an empty list to store the names for saving
save_names = []

# Dictionary to store the means and stds for each variable
statistics = {}

# Iterate through the dictionary, load the files, and append to arrays and save_names
for file_prefix, var_name in var_names.items():
    # Load the array
    array = np.load(f'{subd}/{file_prefix}_1979-2018_5.625deg.npy')
    print(f'Loaded {var_name}')
    
    # Calculate mean and standard deviation
    mean_value = np.mean(array)
    std_value = np.std(array)
    
    # Convert NumPy float32 to native Python float before saving
    statistics[var_name] = {"mean": float(mean_value), "std": float(std_value)}
    
    # Append the array and variable name for saving
    arrays.append(array)
    save_names.append(var_name)

# Combine all loaded arrays along a new axis (e.g., axis 0)
combined_array = np.stack(arrays, axis=0)

# If there is an empty axis, squeeze it and transpose the array
combined_array = np.squeeze(combined_array, 1)
combined_array = np.transpose(combined_array, (1, 0, 2, 3))

# Create the filename by joining the variable names with underscores
combined_filename = "_".join(save_names)

# Save the combined array if needed (optional, commented out)
np.save(f'{subd}/{combined_filename}_1979-2018_5.625deg.npy', combined_array)

# Save the statistics to a JSON file
json_file = f'{subd}/norm_factors.json'
with open(json_file, 'w') as f:
    json.dump(statistics, f, indent=4)

print(f"Normalization factors saved to {json_file}")

# Print out the mean and std for each variable
for var_name, stats in statistics.items():
    print(f"{var_name}: Mean = {stats['mean']}, Std = {stats['std']}")

"""
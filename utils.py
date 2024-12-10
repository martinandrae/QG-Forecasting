import numpy as np
import torch
from torch.utils.data import Sampler

class QGDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path,     # str: Path to the dataset files.
                 dataset_mode,     # str: Dataset dataset_mode ('train', 'val', 'test').
                 sample_counts,    # tuple: Total, training, and validation sample counts (total_samples, train_samples, val_samples).
                 dimensions,        # tuple: Dimensions of the dataset (variables, latitude, longitude).
                 lead_time,        # int: Current lead time for forecasting.
                 max_lead_time,    # int: Maximum lead time we want to forecast.
                 norm_factors,     # tuple: Mean and standard deviation for normalization (mean, std_dev).
                 device,           # torch.device: Device on which tensors will be loaded.
                 lead_time_range,  # Range of lead time
                 spinup = 0,       # int: Number of samples to discard at the start for stability.
                 spacing = 1,      # int: Sample selection interval for data reduction.
                 dtype='float32',   # str: Data type of the dataset (default 'float32').
                 offset=0,          # int: Offset for memory-mapped file (default 0).
                 initial_times=[0,],
                 static_data_path = None,
                 random_lead_time = 0,

                ):
        """
        Initialize a custom Dataset for lazily loading QG samples from a memory-mapped file,
        which allows for efficient data handling without loading the entire dataset into memory.
        """
        self.dataset_path = dataset_path
        self.data_dtype = dtype
        self.device = device
        self.offset = offset

        self.dataset_mode = dataset_mode
        self.n_samples, self.n_train, self.n_val = sample_counts
        self.vars, self.n_lat, self.n_lon = dimensions
        self.max_lead_time = max_lead_time
        self.lead_time = lead_time
        self.spinup = spinup + 24 #min(initial_times) # TODO Remove this hardcoding
        self.spacing = spacing
        self.mean, self.std_dev = norm_factors
        self.kmin, self.kmax, self.d = lead_time_range

        self.static_data_path = static_data_path
        self.static_fields = None

        self.static_vars = 0
        if static_data_path != None:
            self.static_fields = self.load_static_data()      
            self.static_vars = self.static_fields.shape[1]
        
        self.initial_times = initial_times
        self.input_times = self.vars * len(self.initial_times)
        self.output_times = self.vars * (len(self.lead_time) if isinstance(lead_time, (list, tuple, np.ndarray)) else 1)

        self.index_array = self._generate_indices()

        self.mmap = self.create_mmap()

        self.random_lead_time = random_lead_time


    def create_mmap(self):
        """Creates a memory-mapped array for the dataset to facilitate large data handling."""
        return np.memmap(self.dataset_path, dtype=self.data_dtype, mode='r', shape=(self.n_samples, self.vars, self.n_lat, self.n_lon), offset=self.offset)

    def load_static_data(self):
        """Load and normalize static fields."""
        static_fields = np.load(self.static_data_path)

        min_vals = np.min(static_fields, axis=(1,2))
        max_vals = np.max(static_fields, axis=(1,2))

        # Avoid division by zero by ensuring max_vals are not equal to min_vals
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Replace zero range with one to avoid division by zero

        # Apply min-max scaling: (x - min) / (max - min)
        scaled_static_fields = (static_fields - min_vals[:, None, None]) / range_vals[:, None, None]
        #scaled_static_fields = np.expand_dims(scaled_static_fields, axis=0)
        return scaled_static_fields

    def _generate_indices(self):
        """Generates indices for dataset partitioning according to the specified dataset_mode."""
        if self.dataset_mode == 'train':
            start, stop = self.spinup, self.n_train
        elif self.dataset_mode == 'val':
            start, stop = self.spinup + self.n_train, self.n_train + self.n_val
        elif self.dataset_mode == 'test':
            start, stop = self.spinup + self.n_train + self.n_val, self.n_samples

        return np.arange(start, stop - self.max_lead_time)[::self.spacing]

    def set_lead_time(self, lead_time):
        """ Updates the lead time lead_time for generating future or past indices."""
        self.lead_time = lead_time

    def set_lead_time_range(self, lead_time_range):
        self.kmin, self.kmax, self.d = lead_time_range

    def get_lead_time(self):
        if self.random_lead_time:
            return self.kmin + self.d * torch.randint(0, 1 + (self.kmax - self.kmin) // self.d, (1,), device=self.device)[0]
        return self.lead_time

    def __len__(self):
        """Returns the number of samples available in the dataset based on the computed indices."""
        return self.index_array.shape[0]

    def __getitem__(self, idx):
        """Retrieves a sample and its corresponding future or past state from the dataset."""
        start_index = self.index_array[idx]
        lead_times = self.get_lead_time()

        x_index = start_index + self.initial_times
        y_index = start_index + lead_times

        X_sample = self.mmap[x_index, :].astype(self.data_dtype)
        Y_sample = self.mmap[y_index, :].astype(self.data_dtype)

        X_sample = (X_sample - self.mean[None, :, None, None]) / self.std_dev[None, :, None, None]
        Y_sample = (Y_sample - self.mean[None, :, None, None]) / self.std_dev[None, :, None, None]

        X_sample = torch.tensor(X_sample, dtype=torch.float32).view(self.input_times, self.n_lat, self.n_lon)
        Y_sample = torch.tensor(Y_sample, dtype=torch.float32).view(self.output_times, self.n_lat, self.n_lon)

        if self.static_vars != 0:
            X_sample = np.concatenate([X_sample, self.static_fields], axis=0)

        return X_sample, Y_sample, lead_times


def get_uniform_k_dist_fn(kmin, kmax, d):
    """ Create the update function """

    def uniform_k_dist(dataset):
        new_k = kmin + d * np.random.randint(0, 1 + (kmax - kmin) // d)
        dataset.set_lead_time(new_k)
    
    return uniform_k_dist

class DynamicKBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last, k_update_callback, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.k_update_callback = k_update_callback
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        # Shuffle indices at the beginning of each epoch if required
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        batch = []
        for idx in self.indices:
            if len(batch) == self.batch_size:
                self.k_update_callback(self.dataset)  # Update `lead_time` before yielding the batch
                yield batch
                batch = []
            batch.append(idx)
        if batch and not self.drop_last:
            self.k_update_callback(self.dataset)  # Update `lead_time` for the last batch if not dropping it
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

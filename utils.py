import numpy as np
import torch
from torch.utils.data import Sampler

class QGSamplesDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, mode, p_train, k, spinup, spacing, iterations, mean_data, std_data, device, transform=None):
        """
        Custom Dataset for loading QG samples lazily.

        Parameters:
        - data_path (str): Path to the dataset files.
        - mode (str): Mode of the dataset to be loaded ('train', 'val', 'test').
        - p_train (float): Percentage of data to be used for training.
        - k, spinup, spacing, iterations: Parameters for data generation.
        - transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_path = data_path
        self.data_dtype = 'float32'
        self.device = device

        self.mode = mode

        self.p_train = p_train
        self.k = k
        self.spinup = spinup
        self.spacing = spacing
        self.iterations = iterations
        self.mean_data = mean_data
        self.std_data = std_data

        self.transform = transform

        self.total_rows, self.total_columns = self._calculate_dimensions()
        self.X_indices, self.Y_indices = self._generate_indices()

        self.mmap = self.create_mmap()

    
    def create_mmap(self):
        return np.memmap(self.data_path, dtype=self.data_dtype, mode='r', shape=(self.total_rows, self.total_columns))


    def _calculate_dimensions(self):
        total_rows = self.iterations + 1
        total_columns = 4225 
        return total_rows, total_columns


    def _generate_indices(self):
        n_train = int(np.round(self.p_train * (self.total_rows - self.spinup)))  # Number of training samples
        n_val = int(np.round((1 - self.p_train) / 2 * (self.total_rows - self.spinup)))  # Number of validation samples
        # Assuming the remaining samples are used for testing

        if self.mode == 'train':
            start, stop = self.spinup, self.spinup + n_train
        elif self.mode == 'val':
            start, stop = self.spinup + n_train, self.spinup + n_train + n_val
        elif self.mode == 'test':
            start, stop = self.spinup + n_train + n_val, self.total_rows

        fit_x, fit_y = slice(start, stop - self.k), slice(start + self.k, stop)

        # Adjust indices for spacing if necessary
        X_indices = np.arange(fit_x.start, fit_x.stop)[::self.spacing]
        Y_indices = np.arange(fit_y.start, fit_y.stop)[::self.spacing]

        return X_indices, Y_indices


    def __len__(self):
        # Return the length of the dataset
        return self.X_indices.shape[0]

    def __getitem__(self, idx):
        # Calculate the actual data indices for X and Y based on the provided dataset index
        x_index = self.X_indices[idx]
        y_index = self.Y_indices[idx]

        # Access the specific samples directly from the memory-mapped array
        # This operation does not load the entire dataset into memory
        X_sample = self.mmap[x_index, :].astype(self.data_dtype)
        Y_sample = self.mmap[y_index, :].astype(self.data_dtype)

        # Normalize the samples if needed
        X_sample = (X_sample - self.mean_data) / self.std_data
        Y_sample = (Y_sample - self.mean_data) / self.std_data

        # Convert to tensors before returning
        X_sample = torch.tensor(X_sample, dtype=torch.float32).to(self.device)
        Y_sample = torch.tensor(Y_sample, dtype=torch.float32).to(self.device)

        # View the samples as 2D images
        X_sample = X_sample.view(-1, 65, 65)
        Y_sample = Y_sample.view(-1, 65, 65)

        return X_sample, Y_sample
    

class NWPDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, n_val, spacing, device):
        """
        Custom Dataset for loading QG samples lazily.

        Parameters:
        - data_path (str): Path to the dataset files.
        - mode (str): Mode of the dataset to be loaded ('train', 'val', 'test').
        - p_train (float): Percentage of data to be used for training.
        - k, spinup, spacing, iterations: Parameters for data generation.
        - transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_path = data_path
        self.data_dtype = 'float32'
        self.device = device
        
        self.n_val = n_val
        self.kmax=150
        self.total_rows, self.total_columns = self._calculate_dimensions()
        
        self.spacing = spacing
        self.indices = self._generate_indices()

        self.mmap = self.create_mmap()

    def _generate_indices(self):
        return np.arange(0, self.total_rows)[::self.spacing]
    
    def _calculate_dimensions(self):
        total_rows = self.n_val - self.kmax
        total_columns = 4225 
        return total_rows, total_columns
    
    def create_mmap(self):
        return np.memmap(self.data_path, dtype=self.data_dtype, mode='r', shape=(self.total_rows, self.total_columns))

    def __len__(self):
        # Return the length of the dataset
        return self.indices.shape[0]

    def __getitem__(self, idx):
        idx = self.indices[idx]

        # Access the specific samples directly from the memory-mapped array
        # This operation does not load the entire dataset into memory
        X_sample = self.mmap[idx, :].astype(self.data_dtype)

        # Convert to tensors before returning
        X_sample = torch.tensor(X_sample, dtype=torch.float32).to(self.device)

        # View the samples as 2D images
        X_sample = X_sample.view(-1, 65, 65)

        return X_sample 
    


class TimeSampleDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, mode, p_train, kmax, spinup, spacing, iterations, mean_data, std_data, device, transform=None):
        """
        Custom Dataset for loading QG samples lazily.

        Parameters:
        - data_path (str): Path to the dataset files.
        - mode (str): Mode of the dataset to be loaded ('train', 'val', 'test').
        - p_train (float): Percentage of data to be used for training.
        - k, spinup, spacing, iterations: Parameters for data generation.
        - transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_path = data_path
        self.data_dtype = 'float32'
        self.device = device

        self.mode = mode

        self.p_train = p_train
        self.kmax = kmax
        self.k = kmax

        self.spinup = spinup
        self.spacing = spacing
        self.iterations = iterations
        self.mean_data = mean_data
        self.std_data = std_data

        self.transform = transform

        self.total_rows, self.total_columns = self._calculate_dimensions()
        self.X_indices = self._generate_x_indices()

        self.mmap = self.create_mmap()

    
    def create_mmap(self):
        return np.memmap(self.data_path, dtype=self.data_dtype, mode='r', shape=(self.total_rows, self.total_columns))


    def _calculate_dimensions(self):
        total_rows = self.iterations + 1
        total_columns = 4225 
        return total_rows, total_columns


    def _generate_x_indices(self):
        n_train = int(np.round(self.p_train * (self.total_rows - self.spinup)))  # Number of training samples
        n_val = int(np.round((1 - self.p_train) / 2 * (self.total_rows - self.spinup)))  # Number of validation samples
        # Assuming the remaining samples are used for testing

        if self.mode == 'train':
            start, stop = self.spinup, self.spinup + n_train
        elif self.mode == 'val':
            start, stop = self.spinup + n_train, self.spinup + n_train + n_val
        elif self.mode == 'test':
            start, stop = self.spinup + n_train + n_val, self.total_rows

        return np.arange(start, stop - self.kmax)[::self.spacing]

    def set_k(self, k):
        """
        Set the k value for generating Y_indices.
        """
        self.k = k

    def __len__(self):
        # Return the length of the dataset
        return self.X_indices.shape[0]

    def __getitem__(self, idx):
        x_index = self.X_indices[idx]

        y_index = x_index + self.k

        X_sample = self.mmap[x_index, :].astype(self.data_dtype)
        Y_sample = self.mmap[y_index, :].astype(self.data_dtype)

        X_sample = (X_sample - self.mean_data) / self.std_data
        Y_sample = (Y_sample - self.mean_data) / self.std_data

        # Convert to tensors before returning
        X_sample = torch.tensor(X_sample, dtype=torch.float32)#.to(self.device)
        Y_sample = torch.tensor(Y_sample, dtype=torch.float32)#.to(self.device)

        # View the samples as 2D images
        X_sample = X_sample.view(-1, 65, 65)
        Y_sample = Y_sample.view(-1, 65, 65)

        return X_sample, Y_sample, self.k
    

def update_k_per_batch(dataset, kmin, d):
    new_k = kmin + d * np.random.randint(0, 1 + (dataset.kmax-kmin) // d)
    dataset.set_k(new_k)

class DynamicKBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last, k_update_callback, shuffle=False, kmin=50, d=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.k_update_callback = k_update_callback
        self.indices = list(range(len(dataset)))
        
        self.kmin = kmin
        self.d = d

    def __iter__(self):
        # Shuffle indices at the beginning of each epoch if required
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        batch = []
        for idx in self.indices:
            if len(batch) == self.batch_size:
                self.k_update_callback(self.dataset, self.kmin, self.d)  # Update `k` before yielding the batch
                yield batch
                batch = []
            batch.append(idx)
        if batch and not self.drop_last:
            self.k_update_callback(self.dataset, self.kmin, self.d)  # Update `k` for the last batch if not dropping it
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
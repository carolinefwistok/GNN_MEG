import os
import numpy as np
import scipy.io
import h5py
import mne
import matplotlib.pyplot as plt
from mne.time_frequency import psd_array_welch


# Load the .mat file
def load_mat_file(file_path):
    mat = scipy.io.loadmat(file_path)
    return mat

# Extract the scouts data from the .mat file
def extract_scouts_data(mat):
    values = mat['Value'][:]
    descriptions = mat['Description'][:]
    return values, descriptions

# Process the scouts data
def process_scouts_data(values, descriptions):
    scouts_data = {}
    for i, desc in enumerate(descriptions):
        scout_name = str(desc[0])
        scout_signal = values[i, :]
        scouts_data[scout_name] = scout_signal
    return scouts_data

def load_scouts_data(folder_path, file_name, output_file, fif_directory):
    # Load the .mat file and extract the scouts data
    mat = load_mat_file(os.path.join(folder_path, file_name))
    values, descriptions = extract_scouts_data(mat)
    scouts_data = process_scouts_data(values, descriptions)

    # Save the scouts data to an HDF5 file
    with h5py.File(output_file, 'w') as h5f:
        for scout_name, scout_signal in scouts_data.items():
            h5f.create_dataset(scout_name, data=scout_signal)
            print(f'File {output_file} saved successfully!')

    # Print the data for each scout
    for scout_name, scout_signal in scouts_data.items():
        print(f"Scout: {scout_name}, Signal length: {int(scout_signal.shape[0])/2400} seconds")
    
    print(f"Total number of scouts: {len(scouts_data)}")

    # Automatically generate the scout file prefix
    scout_file_prefix = '_'.join(file_name.split('_')[:3])
    fif_file = None

    # Find the corresponding fif file in the fif directory
    for fif_filename in os.listdir(fif_directory):
        print('filename', fif_filename)
        if fif_filename.startswith(scout_file_prefix) and fif_filename.endswith('_FT_data_raw.fif'):
            print('fif filename found!')
            fif_file = os.path.join(fif_directory, fif_filename)
    
    if fif_file is None:
        raise FileNotFoundError(f"No matching FIF file found for scout file prefix {scout_file_prefix}")

    # Load the fif file
    raw = mne.io.read_raw_fif(fif_file, preload=True)

    # Extract sampling frequency and duration
    sfreq = raw.info['sfreq']
    duration = raw.times[-1] - raw.times[0]
    times = np.linspace(0, duration, int(sfreq * duration))

    return scouts_data, sfreq, times

# Define folder path and filename of .mat file
folder_path = r'F:\MEG GNN\GNN\Data\Raw\Scout files\Mat files'
file_name = 'PT10_TONIC_run08_SCOUTS.mat'

# Match the scout file name to an existing fif file
fif_directory = r'F:\MEG GNN\GNN\Data\Raw\Fif files - NOT USED'

# Save the scouts data to an HDF5 file
output_directory = r'F:\MEG GNN\GNN\Data\Raw\Scout files'
output_file = os.path.join(output_directory, file_name.replace('SCOUTS.mat', 'scout_data.h5'))

scouts_data, sfreq, times = load_scouts_data(folder_path, file_name, output_file, fif_directory)

# Calculate the minimum length across all scout_signal arrays and the times array
min_length = min(len(times) - 2000, min(len(scout_signal) for scout_signal in scouts_data.values()))
if min_length < 0:
    min_length = min(len(times), min(len(scout_signal) for scout_signal in scouts_data.values()))

# Ensure the lengths of times and scout_signal match
times = times[:min_length]
for scout_name, scout_signal in scouts_data.items():
    scouts_data[scout_name] = scout_signal[:min_length]

plot = True

if plot:
    # Plot the scout data against the time vector
    plt.figure(figsize=(15, 10))
    for scout_name, scout_signal in scouts_data.items():
        plt.plot(times, scout_signal*10**12, label=scout_name)
    # plt.xlim((times[0], times[-1]-3000))
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    plt.title('Scout Data Over Time')
    plt.legend()
    plt.show()

    # Compute and plot the PSD using Welch method
    win_length = 4  # seconds
    overlap = 0.5  # 50% overlap
    n_samples = int(win_length * sfreq)
    n_overlap = int(n_samples * overlap)

    plt.figure(figsize=(15, 10))
    for scout_name, scout_signal in scouts_data.items():
        # Compute PSD using Welch method
        psd, freqs = psd_array_welch(scout_signal*10**12, sfreq, n_fft=n_samples, n_overlap=n_overlap, average='mean')
        psd_sqrt = np.sqrt(psd)  # Take the square root of the PSD
        
        # Plot the PSD
        plt.plot(freqs, psd_sqrt, label=scout_name)

    plt.xlim((0,300))
    plt.yscale("log")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.title('Power Spectral Density (PSD) of Scout Signals')
    plt.legend()
    plt.show()

def load_scouts_data_from_hdf5(file_path):
    scouts_data = {}
    with h5py.File(file_path, 'r') as f:
        for scout_name in f.keys():
            scouts_data[scout_name] = f[scout_name][:]
    return scouts_data

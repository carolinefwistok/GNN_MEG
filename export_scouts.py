import os
import numpy as np
import scipy.io
import h5py
import mne
from mne.filter import filter_data, resample
import matplotlib.pyplot as plt
from mne.time_frequency import psd_array_welch


def load_mat_file(file_path):
    '''
    Loads the .mat file and returns the object.
    
    INPUT:
        - file_path     : Path to the .mat file
    OUTPUT:
        - mat           : .mat file object
    '''

    mat = scipy.io.loadmat(file_path)
    return mat

def extract_scouts_data(mat):
    '''
    Extracts the scouts data from the .mat file and returns the values and descriptions.
    
    INPUT:
        - mat           : .mat file object
    OUTPUTS:
        - values        : Numpy array containing the scout signals
        - descriptions  : Numpy array containing the scout names
    '''

    values = mat['Value'][:]
    descriptions = mat['Description'][:]
    print(descriptions)
    return values, descriptions

def process_scouts_data(values, descriptions):
    '''
    Processes the scouts data and returns a dictionary containing the scout names and signals.
    
    INPUTS:
        - values        : Numpy array containing the scout signals
        - descriptions  : Numpy array containing the scout names
    
    OUTPUT:
        - scouts_data   : Dictionary containing the scout names and signals
    '''

    scouts_data = {}
    for i, desc in enumerate(descriptions):
        scout_name = str(desc[0])
        scout_signal = values[i, :]
        scouts_data[scout_name] = scout_signal
    return scouts_data

def load_scouts_data(folder_path, file_name, output_file, crop_start_time=None, resample_freq=None):
    '''
    Loads the scouts data from a .mat file, optionally crops it, and saves it to an HDF5 file.

    INPUTS:
        - folder_path      : Path to the folder containing the .mat file
        - file_name        : Name of the .mat file
        - output_file      : Path to save the HDF5 file
        - crop_start_time  : Time (in seconds) to start cropping the signal (optional)
        - resample_freq    : Desired sampling frequency after resampling (optional)


    OUTPUT:
        - scouts_data      : Dictionary containing the cropped scouts data
    '''
    # Load the .mat file and extract the scouts data
    mat = load_mat_file(os.path.join(folder_path, file_name))
    values, descriptions = extract_scouts_data(mat)
    scouts_data = process_scouts_data(values, descriptions)

    # Match the scout file name to an existing FIF file to get sampling frequency
    fif_directory = r'F:\MEG_GNN\GNN\Data\Raw_all'
    sfreq, times = load_fif_file(file_name, fif_directory)

    # Crop the signals if crop_start_time is provided
    if crop_start_time is not None:
        crop_start_index = int(crop_start_time * sfreq)  # Calculate the start index
        times = times[crop_start_index:]  # Crop the times array
        for scout_name, scout_signal in scouts_data.items():
            scouts_data[scout_name] = scout_signal[crop_start_index:]  # Crop each signal

    # Resample the signals if resample_freq is provided
    if resample_freq is not None:
        resample_ratio = resample_freq / sfreq  # Calculate the resampling ratio
        sfreq = resample_freq  # Update the sampling frequency
        for scout_name, scout_signal in scouts_data.items():
            scouts_data[scout_name] = resample(scout_signal, up=resample_ratio, down=1)

        # Update the times array to match the new sampling frequency
        times = np.linspace(times[0], times[-1], len(scouts_data[next(iter(scouts_data))]))

    # Save the scouts data to an HDF5 file
    with h5py.File(output_file, 'w') as h5f:
        for scout_name, scout_signal in scouts_data.items():
            h5f.create_dataset(scout_name, data=scout_signal)
            print(f'File {output_file} saved successfully!')

    # Print the data for each scout
    for scout_name, scout_signal in scouts_data.items():
        print(f"Scout: {scout_name}, Signal length: {len(scout_signal) / sfreq} seconds")
    
    print(f"Total number of scouts: {len(scouts_data)}")

    return scouts_data

def load_fif_file(file_name, fif_directory):
    '''
    Loads the corresponding FIF file for the scout file and returns the sampling frequency and time vector.
    
    INPUTS:
        - file_name      : Name of the scout file
        - fif_directory  : Path to the directory containing the FIF files
        
    OUTPUTS:
        - sfreq          : Sampling frequency of the FIF file
        - times          : Time vector of the FIF file
    '''

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

    return sfreq, times

# Define folder path and filename of .mat file
folder_path = r'F:\MEG_GNN\GNN\Data\Scout_all\Scout_58_MN'
file_names = [f for f in os.listdir(folder_path) if f.endswith('.mat')]

for file_name in file_names:
    print(f"Processing file: {file_name}")
    # Save the scouts data to an HDF5 file
    output_directory = r'F:\MEG_GNN\GNN\Data\Scout_all\Scout_58_MN'
    output_file = os.path.join(output_directory, file_name.replace('SCOUTS.mat', 'scout_data.h5'))

    # Crop signals starting from 105 seconds
    crop_start_time = None  # seconds
    resample_freq = None  # Hz
    scouts_data = load_scouts_data(folder_path, file_name, output_file, crop_start_time=crop_start_time, resample_freq=resample_freq)

    plot = False

    if plot:
        # Match the scout file name to an existing fif file
        fif_directory = r'F:\MEG_GNN\GNN\Data\Raw_all'
        sfreq, times = load_fif_file(file_name, fif_directory)

        # Calculate the minimum length across all scout_signal arrays and the times array
        min_length = min(len(times) - 2000, min(len(scout_signal) for scout_signal in scouts_data.values()))
        if min_length < 0:
            min_length = min(len(times), min(len(scout_signal) for scout_signal in scouts_data.values()))

        # Ensure the lengths of times and scout_signal match
        times = times[:min_length]
        for scout_name, scout_signal in scouts_data.items():
            scouts_data[scout_name] = filter_data(scout_signal[:min_length], sfreq, l_freq=0.1, h_freq=None)

        # Calculate the median and MAD for the entire epoch
        scout_median = [np.median(scout_signal) for scout_signal in scouts_data.values()]
        scout_mad = [np.median(np.abs(epoch_data - epoch_median)) for epoch_data, epoch_median in zip(scouts_data.values(), scout_median)]

        # Plot the scout data against the time vector
        plt.figure(figsize=(15, 10))
        for scout_name, scout_signal in scouts_data.items():
            plt.plot(times, scout_signal*10**12, label=scout_name)
            plt.axhline(y=scout_median[0]*10**12, color='r', linestyle='--', label='Median')
            plt.axhline(y=scout_mad[0]*10**12, color='g', linestyle='--', label='MAD')
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

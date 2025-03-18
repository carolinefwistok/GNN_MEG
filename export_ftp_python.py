import os
import numpy as np
from scipy import io
import h5py
from mne import create_info
from mne.io import RawArray, read_raw_fif
from mne.time_frequency import psd_array_welch
from mne import preprocessing
import matplotlib.pyplot as plt

# Specify the folder path containing .mat files from Fieldtrip
folder_path = r'F:\MEG GNN\GNN\Data\Raw\Export from Brainstorm\FT data'

# Specify output folder path for MNE Python format files
output_folder = r'F:\MEG GNN\GNN\Data\Raw\Fif files - NOT USED'


def load_ft_raw_from_folder(folder_path, var_name):
    '''
    Function to export multiple Fieldtrip data structures to MNE Python format.

    INPUTS:  
        folder_path     : path to the folder containing .mat files
        var_name        : name of variable in .mat files

    OUTPUT:
        raw_dict        : dictionary of MNE raw objects of MEG data with filenames as keys
    '''

    raw_dict = {}  # Dictionary to hold raw objects with filenames as keys

    # Iterate over all files in the specified folder
    for filename in os.listdir(folder_path):
        print(filename)
        if filename.endswith('.mat'):
            # Create output file path for saving the raw object
            output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_raw.fif")

            # Check if the output file already exists
            if os.path.exists(output_file):
                print(f"File {filename} already exists. Skipping saving.")

                # Load the existing .fif file into raw_dict
                raw = read_raw_fif(output_file, preload=True)
                raw_dict[filename] = raw

            # Define the path to the current .mat file and convert to raw object
            else:
                file_path = os.path.join(folder_path, filename)  # Full path to the .mat file
                print(f'Loading {file_path}...')
                raw = load_ft_raw(file_path, var_name)  # Call the function to export the .mat file to raw object
                raw_dict[filename] = raw  # Store the raw object in the dictionary with the filename as the key'

                # Save raw object
                raw.save(
                    fname=output_file,
                    picks='all',  # Save all channels
                    tmin=0,       # Start from the beginning
                    tmax=None,    # Save until the end
                    proj=True,    # Save with projections applied
                    overwrite=True # Overwrite if the file already exists
                )

    return raw_dict

def load_ft_raw(fname, var_name):
    '''
    Function to export Fieldtrip data structure to MNE Python format.

    INPUTS:  
        fname       : path to .mat file
        var_name    : name of variable in .mat file
    
    OUTPUT:
        raw         : MNE raw object of MEG data
    '''
    
    # Load Fieldtrip data structure using H5PY
    with h5py.File(fname, 'r') as f:
        # Open the .mat data
        ft_data_group = f[var_name]
        
        # Load timeseries data from ftData
        avg_data = ft_data_group['avg'][:]
        time_data = ft_data_group['time'][:]
        
        # Load channel labels from ftData/grad/label
        labels = ft_data_group['grad']['label'][:]

        # Load channel types from ftData/grad/chantype
        types = ft_data_group['grad']['chantype'][:]

        # Load channel positions from ftData/grad/chanpos
        chan_pos = ft_data_group['grad']['chanpos'][:][:]
        chan_pos = np.array(chan_pos)
        chan_pos = chan_pos.T
        print('positions', chan_pos.shape)

        # Load channel orientation from ftData/grad/chanori
        chan_ori = ft_data_group['grad']['chanori'][:][:]
        chan_ori = np.array(chan_ori)
        chan_ori = chan_ori.T
        print('orientation', chan_ori.shape)

        # Dereference channel labels if they are references
        chan_names = []
        for ref in labels.flatten():
            if isinstance(ref, h5py.Reference):
                referenced_obj = f[ref]
                if isinstance(referenced_obj, h5py.Dataset):
                    values = referenced_obj[()]
                    if isinstance(values, np.ndarray):
                        channel_name = ''.join(chr(c[0]) for c in values)
                        chan_names.append(channel_name)
            else:
                chan_names.append(ref)

        # Dereference channel types if they are references
        chan_types = []
        for ref in types.flatten():
            if isinstance(ref, h5py.Reference):
                referenced_obj = f[ref]
                if isinstance(referenced_obj, h5py.Dataset):
                    values = referenced_obj[()]
                    if isinstance(values, np.ndarray):
                        channel_type = ''.join(chr(c[0]) for c in values)
                        chan_types.append(channel_type)
            else:
                chan_types.append(ref)

        # Change name to channel type description that is used in MNE
        ch_types = []
        for chan_type in chan_types:
            if chan_type == 'megaxial':
                ch_types.append('mag')  # Change 'megaxial' to 'mag'
            else:
                ch_types.append(chan_type)  # Keep the original type if it's not 'megaxial'

        # Calculate sampling frequency from time data
        sfreq = 1 / (time_data[1] - time_data[0]) 

        # Select only the data corresponding to the selected channels
        data = avg_data[:, [chan_names.index(name) for name in chan_names]] # (in this case, all channels are selected)
        data = data.T  # Transpose data to have shape (n_time, n_chans)

        # Create info structure for MNE
        info = create_info(chan_names, sfreq, ch_types=ch_types)

        # Create loc array for each channel
        for i in range(len(chan_names)):
            loc = np.zeros(12)  # Initialize loc array

            # Channel loc array should consist of nominal channel position [:3], and EX, EY, EZ [3:] normal triplets of the coil orientation matrix
            loc[:3] = chan_pos[i]  # Set nominal channel position

            # Set the orientation matrix [EX EY EZ] using channel orientation
            loc[3:6] = chan_ori[i]  # EX
            
            # Define EY as an orthogonal vector to EX
            if np.all(chan_ori[i] == np.array([0, 0, 1])):  # Special case if EX is [0, 0, 1]
                loc[6:9] = np.array([1, 0, 0])  # Choose an arbitrary orthogonal vector
            else:
                # Calculate EY as a vector that is orthogonal to EX
                ey_candidate = np.array([0, 0, 1])
                loc[6:9] = np.cross(chan_ori[i], ey_candidate)

            # Calculate EZ as the cross product of EX and EY (EZ should be orthogonal to both EX and EY)
            loc[9:12] = np.cross(loc[3:6], loc[6:9])
         
            # Set the loc array in the info structure
            info['chs'][i]['loc'] = loc

        # Create RawArray from the data
        raw = RawArray(data, info)

        # Check if the raw object contains all 46 selected channels
        selected_channels = ['MLO31', 'MLO23', 'MLO34',
                            'MZO02',
                            'MRO31', 'MRO23', 'MRO34',
                            'MLT11', 'MLT23', 'MLT36', 'MLT24',
                            'MLF22', 'MLF43', 'MLF55',
                            'MZF02', 'MZF03', 
                            'MRF22', 'MRF43', 'MRF55',
                            'MRT11', 'MRT23', 'MRT36', 'MRT24',
                            'MRC21', 'MRC23', 'MRC53', 'MRP23',
                            'MLC21', 'MLC23', 'MLC53', 'MLP23',
                            'MLP33', 'MLP44', 'MLP41', 'MLP54', 'MLP57',
                            'MRP33', 'MRP44', 'MRP41', 'MRP54', 'MRP57',
                            'MZP01',
                            'MZC01', 'MZC02', 'MZC03','MZC04']
        
        available_channels = [ch for ch in selected_channels if ch in raw.ch_names]
        print('Number of nodes:', len(available_channels))
        
        # Identify missing channels
        missing_channels = [ch for ch in selected_channels if ch not in available_channels]

        if len(available_channels) != len(selected_channels):
            print(f'File {fname} does not contain all selected MEG channels!')
            print('Missing channels:', missing_channels)
    return raw

# Load raw data objects from the folder
raw_objects = load_ft_raw_from_folder(folder_path, 'ftData')

for filename, raw in raw_objects.items():
    # Check if the raw object contains all 46 selected channels
    selected_channels = ['MLO31', 'MLO23', 'MLO34',
                        'MZO02',
                        'MRO31', 'MRO23', 'MRO34',
                        'MLT11', 'MLT23', 'MLT36', 'MLT24',
                        'MLF22', 'MLF43', 'MLF55',
                        'MZF02', 'MZF03', 
                        'MRF22', 'MRF43', 'MRF55',
                        'MRT11', 'MRT23', 'MRT36', 'MRT24',
                        'MRC21', 'MRC23', 'MRC53', 'MRP23',
                        'MLC21', 'MLC23', 'MLC53', 'MLP23',
                        'MLP33', 'MLP44', 'MLP41', 'MLP54', 'MLP57',
                        'MRP33', 'MRP44', 'MRP41', 'MRP54', 'MRP57',
                        'MZP01',
                        'MZC01', 'MZC02', 'MZC03','MZC04']

    available_channels = [ch for ch in selected_channels if ch in raw.ch_names]
    print('Number of nodes:', len(available_channels))

    # Identify missing channels
    missing_channels = [ch for ch in selected_channels if ch not in available_channels]

    if len(available_channels) != len(selected_channels):
        print(f'File {filename} does not contain all selected MEG channels!')
        print('Missing channels:', missing_channels)

# Plot raw data for all raw objects
plot_raw = False # Set to True to plot the raw objects

if plot_raw:
    for filename, raw in raw_objects.items():
        fig = raw.plot(block=True, scalings='auto')

# Plot PSD for all raw object
plot_psd = False # Set to True to plot the PSD

if plot_psd:
    for filename, raw in raw_objects.items():
        print(f'Plotting file {filename}')
        # Define sampling frequency
        sfreq = raw.info['sfreq']  # Sampling frequency from the raw object

        # Window and overlap
        win_length = 4  # seconds
        overlap = 0.5  # 50% overlap
        n_samples = int(win_length * sfreq)
        n_overlap = int(n_samples * overlap)

        # Get the data from the raw object
        data, times = raw[:, :]  # Get all channels and all time points
        print('Data shape:', data.shape)
        print('Data in T', data)
        data = data * 10**15 # Convert data to femto Tesla (1 fT = 10^-15 T)
        print('Data in fT', data)

        # Compute PSD using Welch method
        psd, freqs = psd_array_welch(data, 
                                    sfreq=sfreq, 
                                    n_fft=n_samples, 
                                    n_overlap=n_overlap, 
                                    average='mean',
                                    remove_dc=True,
                                    output='power')

        # Convert PSD to fT/sqrt(Hz)
        psd_sqrt = np.sqrt(psd)  # Take the square root of the PSD

        # Visualizing the PSD
        plt.plot(freqs, psd_sqrt.T)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density (fT/sqrt(Hz))')
        plt.title('Power Spectral Density for MEG Channels')
        plt.xlim((0,300))
        plt.yscale("log")
        plt.legend()
        plt.grid()
        plt.show()

        print("Printing PSD with different settings...")
        # Resample the raw data
        resample_freq = 256
        raw_resampled = raw.copy().resample(sfreq=resample_freq)

        # Get the data from the resampled raw object
        data_resampled, times_resampled = raw_resampled[:, :]
        print('Resampled data shape:', data_resampled.shape)
        print('Resampled data in T', data_resampled)
        print('Resampled data in fT', data_resampled * 10**15)

        # Compute PSD using Welch method
        psd_resampled, freqs_resamples = psd_array_welch(data_resampled*10**15, 
                                                        sfreq=sfreq, 
                                                        n_fft=n_samples, 
                                                        n_overlap=n_overlap, 
                                                        average='mean')
                                                        # remove_dc=True,
                                                        # output='power')

        # Normalize the PSD for resampled data
        psd_resampled_normalized = psd_resampled / np.sum(psd_resampled, axis=1, keepdims=True)  # Normalize by the total power

        # Convert PSD to fT/sqrt(Hz)
        psd_resampled_sqrt = np.sqrt(psd_resampled_normalized)  # Take the square root of the normalized PSD

        print('freqs', freqs_resamples)
        freqs_resamples = np.linspace(0, 256, psd_resampled_sqrt.shape[1])
        print('freqs', freqs_resamples)
        # Visualizing the PSD
        plt.plot(freqs_resamples, psd_resampled_sqrt.T)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density (fT/sqrt(Hz))')
        plt.title('Power Spectral Density for MEG Channels')
        plt.yscale("log")
        plt.legend()
        plt.grid()
        plt.show()

# for filename, raw in raw_objects.items():
#     noisy_chs, flat_chs, scores = preprocessing.find_bad_channels_maxwell(raw, 
#                                                                           cross_talk=None, 
#                                                                           calibration=None,
#                                                                           return_scores=True,
#                                                                           coord_frame="meg", # mne has a method to transfer CTF's meg to head coord
#                                                                           verbose=True)
    
#     print(f'Noisy channels for {filename}: {noisy_chs}')
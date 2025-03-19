import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from dataset import MEGGraphs
from data_utils import get_stimulation_info

def load_raw_data(file_path):
    '''
    Loads the MEG data as fif-file and picks the selected channels from each brain region.

    INPUT: 
        - file_path      : path to the data file currently being processed
    
    OUTPUT:
        - raw            : raw object of time series data of selected channels
    '''

    # Load raw data file
    raw = mne.io.read_raw_fif(file_path, preload=False)
    
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
        
    # Check which channels are available in the raw data
    available_channels = [ch for ch in selected_channels if ch in raw.ch_names]
    print('Number of nodes:', len(available_channels))
    
    # Pick only the time series of the defined channels 
    raw.pick_channels(available_channels)
    return raw

def create_epochs(raw, stim_info, ramp_time):
    '''
    Creates 60-second epochs based on the events defined in 'create_events'. 
    Stimulation was turned ON in the 60 seconds prior to the event sample; stimulation was turned OFF in the 60 seconds following the event sample.

    INPUTS:
        - raw           : raw object of time series data of central channels
        - stim_info     : dictionary with information about the stimulation times
        - ramp_time     : time in seconds that is removed in the beginning and end of each epoch (to account for ramping of stimulation effects)

    OUTPUT:
        - epochs_stim   : Epochs object containing 60-second stimulation ON epochs
        - epochs_non_stim : Epochs object containing 60-second stimulation OFF epochs
    '''

    # Print raw data information
    print(f'Raw data info: {raw.info}')
    print(f'Raw data shape: {raw.get_data().shape}')

    # Define events stating when stimulation did or did not take place
    events, event_id = MEGGraphs.create_events(None, stim_info)
    print(f'Events: {events}')
    print(f'Event ID: {event_id}')

    # Create epochs during which stimulation took place
    epochs_stim = MEGGraphs.create_epochs(None, raw, events, event_id, ramp_time, label='stim')
    print(f'Number of stimulation ON epochs before dropping bad: {len(epochs_stim)}')
    epochs_stim.drop_bad()
    print(f'Number of stimulation ON epochs after dropping bad: {len(epochs_stim)}')

    # Create epochs during which no stimulation took place
    epochs_non_stim = MEGGraphs.create_epochs(None, raw, events, event_id, ramp_time, label='non_stim')

    return epochs_stim, epochs_non_stim

def split_epochs(epochs, duration, overlap, ramp_time, sfreq, label, threshold):
    '''
    Splits each epoch into subepochs of initialized duration and overlap and returns a list of all subepochs.

    INPUTS:
        - epochs            : mne.Epochs object containing 60-second epochs
        - duration          : duration of final epochs in seconds
        - overlap           : overlap between final epochs in seconds
        - ramp_time         : time in seconds that is removed in the beginning and end of each epoch (to account for ramping of stimulation effects)
        - sfreq             : sampling frequency of the data
        - label             : string defining whether this epoch is for stimulation ON or OFF
        - threshold         : threshold for detecting bad segments

    OUTPUT: 
        - subepochs_list    : list of subepochs of length 'duration'
        - bad_subepochs_list: list of bad subepochs of length 'duration'
    '''

    # Define epoch length
    length_epoch = 60 - 2 * ramp_time

    # Define empty list to fill with subepochs
    subepochs_list = []
    bad_subepochs_list = []
    subepochs_per_epoch = []
    total_subepochs = 0
    print('label', label)

    # Duration is not corresponding to the epoch length, so cropping is needed
    if duration != length_epoch:
        # Since tmin and tmax are different for stimulation ON and OFF epochs, these need to be split    
        if label == 'stim':
            # First subepoch needs to start at -60 seconds, excluding the ramp time
            start = 0  # This is the earliest start time

            # Last subepoch needs to start at 'duration' before the ramp time at the end of the epoch
            stop = int(60 - duration)  # This is the latest start time

            # Ensure that the last tmin does not exceed the minimum required value
            max_last_tmin = 60 - 2* ramp_time - duration
            if stop > max_last_tmin:
                stop = max_last_tmin
            
            # Calculate how many subepochs you will get out of the 60 seconds based on 'duration' and 'overlap' 
            num = int((length_epoch - duration) / (duration - overlap) + 1)
            print('num', num)

            # Create list of all start time points of subepochs
            all_tmin = np.linspace(start, stop, num)

            # Create list of all end time points of subepochs 
            all_tmax = all_tmin + duration
            print('all_tmin', all_tmin)
            print('all_tmax', all_tmax)

            # All subepochs will have the same event sample, but this is not possible in an Epochs object
            # Therefore, create a list of unique (even) numbers to make each event sample unique
            unique_event_samples = list(range(0, 2*num, 2))

        elif label == 'non-stim':
            # First subepoch needs to start at ramp time
            start = 0

            # Last subepoch needs to start at 'duration' before 60
            stop = int(60 - 2*ramp_time - duration)

            # Calculate how many subepochs you will get out of the 60 seconds based on 'duration' and 'overlap'
            num = int((length_epoch - duration) / (duration - overlap) + 1)

            # Create list of all start time points of subepochs
            all_tmin = np.linspace(start, stop, num)

            # Create list of all end time points of subepochs 
            all_tmax = all_tmin + duration
            print('all_tmin', all_tmin)
            print('all_tmax', all_tmax)

            # Create list of OTHER unique (odd) numbers to make each event sample unique
            unique_event_samples = list(range(1, 2*num, 2))
        
        # Iterate over all epochs
        for idx, _ in enumerate(range(len(epochs))):
            subepoch_count = 0 

            # Load data from epoch
            epoch = epochs[idx].get_data() * 10**15  # Scale if necessary

            # Calculate the median and MAD for the entire epoch
            epoch_median = np.median(epoch)
            epoch_mad = np.median(np.abs(epoch - epoch_median))

            # Iterate over all tmin and tmax
            for i, (tmin, tmax) in enumerate(zip(all_tmin, all_tmax)):
                # Load data from epoch
                epoch = epochs[idx].get_data()*10**15
                print('tmin', tmin)
                print('tmax', tmax)
                print('index', total_subepochs)

                # Crop epoch with tmin and tmax
                start_sample = int(tmin * 2400)
                stop_sample = int(tmax * 2400)
                subepoch_data = epoch[:, :, start_sample:stop_sample]

                # Calculate the median of the subepoch
                subepoch_median = np.median(subepoch_data)

                # Calculate the maximum deviation from the epoch median
                max_deviation = np.max(np.abs(subepoch_data - epoch_median))

                # Calculate deviation score based on maximum deviation and MAD
                dev_score = max_deviation / epoch_mad if epoch_mad != 0 else np.inf  # Avoid division by zero
 
                print(f'epoch_median={epoch_median}, subepoch_median={subepoch_median}, max_deviation={max_deviation}, deviation score={dev_score}, threshold={threshold}')
                
                if dev_score > threshold:
                    print(f'Removing bad subepoch: tmin={tmin}, tmax={tmax}')
                    bad_subepochs_list.append((total_subepochs, subepoch_count, mne.EpochsArray(subepoch_data, epochs.info, events=subepoch_events)))
                    # continue  # Uncomment this line if you want to remove the bad subepochs

                # Create unique event sample
                subepoch_events = epochs[idx].events.copy()
                subepoch_events[:, 0] = subepoch_events[:, 0] + unique_event_samples[i]

                # Create subepoch as an mne.EpochsArray object
                subepoch = mne.EpochsArray(subepoch_data, epochs.info, events=subepoch_events)

                # Add subepoch to list
                subepochs_list.append(subepoch)
                subepoch_count += 1
                total_subepochs += 1
            
            subepochs_per_epoch.append(subepoch_count)

    # Duration is corresponding to the epoch length, so no cropping is needed
    else:       
        # Iterate over all epochs
        for idx, _ in enumerate(range(len(epochs))):
            # Load data from epoch
            epoch = epochs[idx].get_data()

            # Ensure epoch data is 3D
            epoch_data = epoch[np.newaxis, ...]

            # Make sure stimulation OFF epochs have a slightly different event sample than stimulation ON epochs
            if label == 'non_stim':
                epochs[idx].events[:, 0] = epochs[idx].events[:, 0] + 1 
            
            # Create subepoch as an mne.EpochsArray object
            subepoch = mne.EpochsArray(epoch_data, epochs.info, events=epochs[idx].events)

            # Add subepoch to list
            subepochs_list.append(subepoch)
            subepoch_count += 1
            total_subepochs += 1

            subepochs_per_epoch.append(subepoch_count)

    return subepochs_list, bad_subepochs_list

def plot_epochs(subepochs_stim, subepochs_non_stim, bad_subepochs_stim, bad_subepochs_non_stim, title):
    '''
    Plots the given epochs using MNE's plotting functions.

    INPUT:
        - subepochs_stim: List of Epochs objects for stimulation ON to be plotted
        - subepochs_non_stim: List of Epochs objects for stimulation OFF to be plotted
        - bad_subepochs_stim: List of bad subepochs for stimulation ON
        - bad_subepochs_non_stim: List of bad subepochs for stimulation OFF
        - title: Title of the plot
    OUTPUT: N/A
    '''
    # Extract the data and events from the list of epochs
    all_data_stim = [epochs.get_data() for epochs in subepochs_stim]
    all_events_stim = [epochs.events for epochs in subepochs_stim]
    all_data_non_stim = [epochs.get_data() for epochs in subepochs_non_stim]
    all_events_non_stim = [epochs.events for epochs in subepochs_non_stim]

    # Combine the data and events from all epochs
    combined_data_stim = np.concatenate(all_data_stim, axis=0)
    combined_events_stim = np.concatenate(all_events_stim, axis=0)
    combined_data_non_stim = np.concatenate(all_data_non_stim, axis=0)
    combined_events_non_stim = np.concatenate(all_events_non_stim, axis=0)

    # Use the info from one of the original epochs
    info = subepochs_stim[0].info

    # Create the combined Epochs object for stimulation ON
    combined_epochs_stim = mne.EpochsArray(combined_data_stim, 
                                           info, 
                                           events=combined_events_stim)

    # Create the combined Epochs object for stimulation OFF
    combined_epochs_non_stim = mne.EpochsArray(combined_data_non_stim, 
                                               info, 
                                               events=combined_events_non_stim)

    # Calculate the median and MAD for the entire combined epochs
    epoch_median_stim = np.median(combined_data_stim)
    epoch_mad_stim = np.median(np.abs(combined_data_stim - epoch_median_stim))

    epoch_median_non_stim = np.median(combined_data_non_stim)
    epoch_mad_non_stim = np.median(np.abs(combined_data_non_stim - epoch_median_non_stim))

    # Plot the epochs in subplots
    num_epochs = max(len(subepochs_stim), len(subepochs_non_stim))
    num_cols = 5  # Number of columns for subplots
    num_rows = (num_epochs + num_cols - 1) // num_cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 15), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(subepochs_stim):
            epoch_data_stim = subepochs_stim[i].get_data()
            avg_epoch_data_stim = np.mean(epoch_data_stim, axis=1).squeeze()  # Average over all channels
            times = subepochs_stim[i].times
            ax.plot(times, avg_epoch_data_stim.T, color='blue', label='Stimulation ON')

            # Plot median and MAD for the entire epoch
            ax.axhline(epoch_median_stim, color='green', linestyle='--', label='Median (ON)')
            ax.axhline(epoch_median_stim + epoch_mad_stim, color='orange', linestyle='--', label='MAD (ON)')
            ax.axhline(epoch_median_stim - epoch_mad_stim, color='orange', linestyle='--')

            # Highlight bad subepochs within the current subepoch
            for total_idx, subepoch_idx, bad_epoch in bad_subepochs_stim:
                if total_idx == i:
                    ax.axvspan(bad_epoch.times[0], bad_epoch.times[-1], color='blue', alpha=0.2, label='Bad Subepoch (ON)')

        if i < len(subepochs_non_stim):
            epoch_data_non_stim = subepochs_non_stim[i].get_data()
            avg_epoch_data_non_stim = np.mean(epoch_data_non_stim, axis=1).squeeze()  # Average over all channels
            times = subepochs_non_stim[i].times
            ax.plot(times, avg_epoch_data_non_stim.T, color='red', label='Stimulation OFF')
            
            # Plot median and MAD for the entire epoch
            ax.axhline(epoch_median_non_stim, color='cyan', linestyle='--', label='Median (OFF)')
            ax.axhline(epoch_median_non_stim + epoch_mad_non_stim, color='yellow', linestyle='--', label='MAD (OFF)')
            ax.axhline(epoch_median_non_stim - epoch_mad_non_stim, color='yellow', linestyle='--')

            # Highlight bad subepochs within the current subepoch
            for total_idx, subepoch_idx, bad_epoch in bad_subepochs_non_stim:
                if total_idx == i:
                    ax.axvspan(bad_epoch.times[0], bad_epoch.times[-1], color='red', alpha=0.2, label='Bad Subepoch (OFF)')

        ax.set_title(f'Epoch {i+1}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Signal (fT)')
        ax.legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_single_epoch_time_series(subepochs, subepoch_index, title):
    '''
    Plots the time series for a single subepoch, including the median, MAD, and max deviation
    calculated from the entire epoch.

    INPUT:
        - subepochs: List of subepoch Epochs objects
        - subepoch_index: Index of the subepoch to be plotted
        - title: Title of the plot
    OUTPUT: N/A
    '''

    # Check if the subepoch_index is valid
    if subepoch_index < 0 or subepoch_index >= len(subepochs):
        raise ValueError("Invalid subepoch index.")

    # Get the selected subepoch
    subepoch = subepochs[subepoch_index]
    subepoch_data = subepoch.get_data()
    avg_subepoch_data = np.mean(subepoch_data, axis=1).squeeze()  # Average over all channels
    times = subepoch.times

    # Get data for the entire epoch (assuming all subepochs belong to the same epoch)
    entire_epoch_data = np.concatenate([epoch.get_data() for epoch in subepochs], axis=0)
    avg_entire_epoch_data = np.mean(entire_epoch_data, axis=1).squeeze()  # Average over all channels

    # Calculate median, MAD, and max deviation for the entire epoch
    epoch_median = np.median(avg_entire_epoch_data)
    epoch_mad = np.median(np.abs(avg_entire_epoch_data - epoch_median))
    max_deviation = np.max(np.abs(avg_subepoch_data - epoch_median))
    dev_score = (max_deviation - epoch_median) / epoch_mad if epoch_mad != 0 else np.inf  # Avoid division by zero

    plt.figure(figsize=(10, 5))
    plt.plot(times, avg_subepoch_data.T, color='blue', label='Subepoch Time Series')
    plt.axhline(epoch_median, color='green', linestyle='--', label='Median (Entire Epoch)')
    plt.axhline(epoch_median + epoch_mad, color='orange', linestyle='--', label='MAD (Entire Epoch)')
    plt.axhline(epoch_median - epoch_mad, color='orange', linestyle='--')
    plt.axhline(epoch_median + max_deviation, color='red', linestyle='--', label='Max Deviation (Subepoch)')
    plt.axhline(epoch_median - max_deviation, color='red', linestyle='--')
    plt.axhline(dev_score, color='purple', linestyle='--', label=f'Deviation Score = {dev_score}')

    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Signal (fT)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def compute_psd(epoch, sfreq, fmax):
    '''
    Calculates the Power Spectral Density (PSD) for each of the selected channels.

    INPUTS: 
        - epoch         : the epoch currently being processed
        - sfreq         : the defined resampling frequency
        - fmax          : the defined maximum frequency for the PSD calculation
    
    OUTPUT:
        - psd           : PSD values for the epoch
        - freqs         : Frequency values corresponding to the PSD
    '''

    psd, freqs = mne.time_frequency.psd_array_welch(epoch.get_data(),
                                                    fmin=1,
                                                    fmax=fmax, 
                                                    sfreq=sfreq,
                                                    n_fft=sfreq,
                                                    average='mean',
                                                    remove_dc=True,
                                                    output='power')
    psd_sqrt = np.sqrt(psd)
    psd = np.squeeze(psd_sqrt)
    return psd, freqs

def z_score_psd(psd_list):
    '''
    Applies z-scoring to the PSD values.

    INPUT:
        - psd_list: List of PSD values to be z-scored (shape: [num_epochs, num_channels, num_frequencies])
    OUTPUT:
        - z_scored_psd_list: List of z-scored PSD values (shape: [num_epochs, num_channels, num_frequencies])
    '''
    psd_array = np.array(psd_list)
    mean_psd = np.mean(psd_array, axis=0)
    std_psd = np.std(psd_array, axis=0)
    z_scored_psd_list = (psd_array - mean_psd) / std_psd
    return z_scored_psd_list

def baseline_correct_psd_on(psd_on_list, mean_psd_off):
    '''
    Applies baseline correction to the PSD values using the mean of the stimulation OFF PSDs.

    INPUT:
        - psd_on_list: List of PSD values for stimulation ON (shape: [num_epochs, num_channels, num_frequencies])
        - mean_psd_off: Mean PSD values for stimulation OFF (shape: [num_channels, num_frequencies])
    OUTPUT:
        - corrected_psd_list: List of baseline-corrected PSD values (shape: [num_epochs, num_channels, num_frequencies])
    '''
    corrected_psd_list = [(psd_on - mean_psd_off) / mean_psd_off for psd_on in psd_on_list]

    # min_value = min([psd.min() for psd in corrected_psd_list])
    # if min_value < 0:
    #     shift_value = abs(min_value)
    #     for psd in corrected_psd_list:
    #         psd += shift_value

    return corrected_psd_list

def baseline_correct_psd_off(psd_off_list, mean_psd_off):
    '''
    Applies baseline correction to the PSD values using the mean of the stimulation OFF PSDs.

    INPUT:
        - psd_off_list: List of PSD values for stimulation OFF (shape: [num_epochs, num_channels, num_frequencies])
        - mean_psd_off: Mean PSD values for stimulation OFF (shape: [num_channels, num_frequencies])
    OUTPUT:
        - corrected_psd_list: List of baseline-corrected PSD values (shape: [num_epochs, num_channels, num_frequencies])
    '''
    corrected_psd_list = [(psd_off - mean_psd_off) / mean_psd_off for psd_off in psd_off_list]

    # min_value = min([psd.min() for psd in corrected_psd_list])
    # if min_value < 0:
    #     shift_value = abs(min_value)
    #     for psd in corrected_psd_list:
    #         psd += shift_value

    return corrected_psd_list

def plot_avg_psd(psd_stim_list, psd_non_stim_list, fmax, title, channel_names):
    '''
    Plots the average Power Spectral Density (PSD) for each channel in a subplot.

    INPUT:
        - psd_stim_list: List of PSD values for stimulation ON to be averaged (shape: [num_epochs, num_channels, num_frequencies])
        - psd_non_stim_list: List of PSD values for stimulation OFF to be averaged (shape: [num_epochs, num_channels, num_frequencies])
        - fmax: Maximum frequency of the PSD computation
        - title: Title of the plot
        - channel_names: List of channel names
    OUTPUT: N/A
    '''

    # Compute the average and standard deviation of the PSD values over all epochs
    avg_psd_stim = np.mean(psd_stim_list, axis=0)
    std_psd_stim = np.std(psd_stim_list, axis=0)

    # Compute the average and standard deviation of the PSD values over all epochs
    avg_psd_non_stim = np.mean(psd_non_stim_list, axis=0)
    std_psd_non_stim = np.std(psd_non_stim_list, axis=0)
    
    num_channels = avg_psd_stim.shape[0]
    num_cols = 6  # Number of columns for subplots
    num_rows = (num_channels + num_cols - 1) // num_cols  # Calculate the number of rows needed
    
    freqs = np.linspace(1, fmax, avg_psd_stim.shape[1])

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 15), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < num_channels:
            ax.plot(freqs, avg_psd_stim[i], label='Stimulation ON', color='blue')
            ax.fill_between(freqs, avg_psd_stim[i] - std_psd_stim[i], avg_psd_stim[i] + std_psd_stim[i], color='blue', alpha=0.2)
            ax.plot(freqs, avg_psd_non_stim[i], label='Stimulation OFF', color='red')
            ax.fill_between(freqs, avg_psd_non_stim[i] - std_psd_non_stim[i], avg_psd_non_stim[i] + std_psd_non_stim[i], color='red', alpha=0.2)
            ax.set_title(channel_names[i])
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('PSD')
            # ax.set_yscale('log')
        else:
            ax.axis('off')  # Turn off unused subplots

    fig.suptitle(title)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_all_epochs_avg(psd_stim_corrected, psd_non_stim_corrected, freqs, title, channel_names):
    '''
    Plots the average Power Spectral Density (PSD) for each epoch in a single subplot figure.

    INPUT:
        - psd_stim_corrected: Corrected PSD values for stimulation ON (shape: [num_epochs, num_channels, num_frequencies])
        - psd_non_stim_corrected: Corrected PSD values for stimulation OFF (shape: [num_epochs, num_channels, num_frequencies])
        - freqs: Frequency values corresponding to the PSD (shape: [num_frequencies])
        - title: Title of the plot
        - channel_names: List of channel names
    OUTPUT: N/A
    '''
    # Compute the average and standard deviation of the PSD values over all epochs
    avg_psd_epochs_stim = np.mean(psd_stim_corrected, axis=1)
    std_psd_epochs_stim = np.std(psd_stim_corrected, axis=1)
    avg_psd_epochs_non_stim = np.mean(psd_non_stim_corrected, axis=1)
    std_psd_epochs_non_stim = np.std(psd_non_stim_corrected, axis=1)
    
    num_stim_epochs = avg_psd_epochs_stim.shape[0]
    num_non_stim_epochs = avg_psd_epochs_non_stim.shape[0]
    num_epochs = max(num_stim_epochs, num_non_stim_epochs)
    num_cols = 6  # Number of columns for subplots
    num_rows = (num_epochs + num_cols - 1) // num_cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 15), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < num_epochs:
            if i < num_stim_epochs:
                ax.plot(freqs, avg_psd_epochs_stim[i], color='blue', label='Stimulation ON')
                ax.fill_between(freqs, avg_psd_epochs_stim[i] - std_psd_epochs_stim[i], avg_psd_epochs_stim[i] + std_psd_epochs_stim[i], color='blue', alpha=0.2)
            if i < num_non_stim_epochs:
                ax.plot(freqs, avg_psd_epochs_non_stim[i], color='red', label='Stimulation OFF')
                ax.fill_between(freqs, avg_psd_epochs_non_stim[i] - std_psd_epochs_non_stim[i], avg_psd_epochs_non_stim[i] + std_psd_epochs_non_stim[i], color='red', alpha=0.2)
            ax.set_title(f'Epoch {i+1}')
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('PSD')
        else:
            ax.axis('off')  # Turn off unused subplots

    fig.suptitle(title)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_single_epoch_psd(epoch_on, epoch_off, mean_psd_off, sfreq, fmax, channel_index, title):
    '''
    Plots the PSD before and after baseline correction for a single channel and a single epoch.

    INPUT:
        - epoch: The epoch to be plotted
        - mean_psd_off: Mean PSD values for stimulation OFF (shape: [num_channels, num_frequencies])
        - sfreq: Sampling frequency
        - fmax: Maximum frequency for the PSD calculation
        - channel_index: Index of the channel to be plotted
        - title: Title of the plot
    OUTPUT: N/A
    '''
    psd_on, freqs = compute_psd(epoch_on, sfreq, fmax)
    psd_corrected_on = (psd_on - mean_psd_off) / mean_psd_off

    psd_off, freqs = compute_psd(epoch_off, sfreq, fmax)
    psd_corrected_off = (psd_off - mean_psd_off) / mean_psd_off

    plt.figure(figsize=(10, 5))
    plt.plot(freqs, psd_corrected_on[channel_index], label='PSD stim ON', color='blue')
    plt.plot(freqs, psd_corrected_off[channel_index], label='PSD stim OFF', color='red')

    plt.title(title)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    '''
    Main function to load, process, and plot the data.

    INPUT: N/A
    OUTPUT: N/A
    '''

    # Define the directory containing the raw data
    directory = r'F:\MEG GNN\GNN\Data\Raw'
    filenames = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.fif')]

    # Define the path to the stimulation information Excel file
    stim_excel_file = r'F:\MEG GNN\MEG data\MEG_PT_notes.xlsx'

    # Define parameters
    duration = 12
    overlap = 4
    ramp_time = 5
    fmin = 1
    fmax = 100
    sfreq = 256
    theshold = 100

    # Initialize the total number of subepochs
    total_epochs = 0

    # List to store the number of subepochs for each file 
    epochs_per_file = []

    # List to store the number of subepochs with stim ON and OFF for each file 
    stim_on_per_file = []
    stim_off_per_file = []

    # Loop through each file
    for file_path in filenames:
        print(f'Processing file: {file_path}')

        # Load raw data
        raw = load_raw_data(file_path)

        # Uncomment the following to plot the raw time series per file
        # raw = raw.resample(sfreq=sfreq)
        # raw.plot(n_channels=32, scalings='auto', title=f'Raw Data {file_path}')

        # Compute and plot PSD for the entire raw data
        data, times = raw[:,:]
        win_length = 4  # seconds
        win_overlap = 0.5  # 50% overlap
        n_samples = int(win_length * raw.info['sfreq'])
        n_overlap = int(n_samples * win_overlap)
        psd_raw, freqs_raw = mne.time_frequency.psd_array_welch(data*10**15,
                                                                fmin=fmin,
                                                                fmax=fmax,
                                                                sfreq=int(raw.info['sfreq']),
                                                                n_fft=n_samples,
                                                                n_overlap=n_overlap,
                                                                average='mean',
                                                                remove_dc=True,
                                                                output='power')
        plt.figure(figsize=(10, 5))
        plt.plot(freqs_raw, np.sqrt(psd_raw).T, label=raw.info['ch_names'])
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [fT/sqrt(Hz)]')
        plt.title(f'PSD of Raw Data {file_path}')
        plt.legend()
        plt.show()

        # Retrieve stimulation info for the current file
        stim_info = get_stimulation_info(stim_excel_file, os.path.basename(file_path))
        print('Stimulation info:', stim_info)

        # Create epochs
        epochs_stim, epochs_non_stim = create_epochs(raw, stim_info, ramp_time)
        print('Number of stimulation ON epochs:', len(epochs_stim))
        print('Number of stimulation OFF epochs:', len(epochs_non_stim))

        # Calculate the number of subepochs
        num_stim_on_epochs = len(epochs_stim)
        num_stim_off_epochs = len(epochs_non_stim)

        # Calculate total amount of subepochs based on duration and overlap
        length_epoch = 60 - 2 * ramp_time  # Specify the length of one epoch
        if overlap == 0:
            epochs_for_file = int(length_epoch / duration) * (num_stim_on_epochs + num_stim_off_epochs)
            num_stim_on_subepochs = int(length_epoch / duration) * num_stim_on_epochs
            num_stim_off_subepochs = int(length_epoch / duration) * num_stim_off_epochs
        else:
            epochs_for_file = int((length_epoch - duration) / (duration - overlap) + 1) * (num_stim_on_epochs + num_stim_off_epochs)
            num_stim_on_subepochs = int((length_epoch - duration) / (duration - overlap) + 1) * num_stim_on_epochs
            num_stim_off_subepochs = int((length_epoch - duration) / (duration - overlap) + 1) * num_stim_off_epochs

        # Calculate amount of subepochs with stimulation ON and amount of subepochs with stimulation OFF
        epochs_per_file.append(epochs_for_file)  # Append the number of subepochs for this file to list
        total_epochs += epochs_for_file  # Update the total number of subepochs
        stim_on_per_file.append(num_stim_on_subepochs)  # Append number of subepochs for this file with stimulation ON
        stim_off_per_file.append(num_stim_off_subepochs)  # Append number of subepochs for this file with stimulation OFF

        print(f'Number of subepochs for file {file_path}: {epochs_for_file}')
        print(f'Number of stimulation ON subepochs: {num_stim_on_subepochs}')
        print(f'Number of stimulation OFF subepochs: {num_stim_off_subepochs}')

        # Split epochs into subepochs
        subepochs_stim, bad_subepochs_stim = split_epochs(epochs_stim, duration, overlap, ramp_time, sfreq, label='stim', threshold=theshold)
        subepochs_non_stim, bad_subepochs_non_stim = split_epochs(epochs_non_stim, duration, overlap, ramp_time, sfreq, label='non-stim', threshold=theshold)
        
        # Resample epochs using specified sampling frequency
        subepochs_stim_resampled = [epoch.resample(sfreq=sfreq) for epoch in subepochs_stim]
        subepochs_non_stim_resampled = [epoch.resample(sfreq=sfreq) for epoch in subepochs_non_stim]
        bad_subepochs_stim_resampled = [(total_idx, subepoch_idx, epoch.resample(sfreq=sfreq)) for total_idx, subepoch_idx, epoch in bad_subepochs_stim]
        bad_subepochs_non_stim_resampled = [(total_idx, subepoch_idx, epoch.resample(sfreq=sfreq)) for total_idx, subepoch_idx, epoch in bad_subepochs_non_stim]

        print('Number of stimulation ON subepochs:', len(subepochs_stim))
        print('Number of stimulation OFF subepochs:', len(subepochs_non_stim))
        print('Subepoch', subepochs_stim)
        print('Bad subepoch list', bad_subepochs_stim)
        
        # Concatenate epochs
        all_data_stim = np.concatenate([epoch.get_data() for epoch in subepochs_stim], axis=0)
        all_data_non_stim = np.concatenate([epoch.get_data() for epoch in subepochs_non_stim], axis=0)

        # Check the sampling rate
        sampling_rate = int(subepochs_stim[0].info['sfreq'])
        print(f'Sampling Rate: {sampling_rate} Hz')

        # Compute PSD of concatenated epochs
        conc_psd_stim, freqs_conc = mne.time_frequency.psd_array_welch(all_data_stim, sfreq=sampling_rate, fmin=fmin, fmax=fmax, n_fft=sampling_rate, average='mean')
        conc_psd_non_stim, freqs_conc = mne.time_frequency.psd_array_welch(all_data_non_stim, sfreq=sampling_rate, fmin=fmin, fmax=fmax, n_fft=sampling_rate, average='mean')
        
        print('shape concatenated epochs', conc_psd_stim.shape)
        print('freqs', freqs_conc)
        # Plot PSD of concatenated epochs
        plt.figure()
        plt.plot(freqs_conc, np.sqrt(conc_psd_stim[0]).T)
        plt.plot(freqs_conc, np.sqrt(conc_psd_non_stim[0]).T)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD')
        plt.title('PSD of one subepochs')
        plt.show()

        plt.figure()
        plt.plot(freqs_conc, np.mean(np.sqrt(conc_psd_stim), axis=0).T, label=f"Stim ON {raw.info['ch_names']}")
        plt.plot(freqs_conc, np.mean(np.sqrt(conc_psd_non_stim), axis=0).T, label=f"Stim OFF {raw.info['ch_names']}")
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD')
        plt.title('PSD of averaged subepochs')
        plt.show()

        # Plot subepochs
        plot_epochs(subepochs_stim_resampled, subepochs_non_stim_resampled, bad_subepochs_stim_resampled, bad_subepochs_non_stim_resampled, f'Stimulation ON and OFF Subepochs for file {file_path}')

        # Compute PSD for each subepoch and store in lists
        combined_psd_list = []
        labels = []

        for epoch in subepochs_stim:
            psd, freqs = compute_psd(epoch, sfreq, fmax)
            combined_psd_list.append(psd)
            labels.append('stim')

        for epoch in subepochs_non_stim:
            psd, freqs = compute_psd(epoch, sfreq, fmax)
            combined_psd_list.append(psd)
            labels.append('non_stim')

        # Apply z-scoring to the PSDs
        # combined_psd_list = z_score_psd(combined_psd_list)

        # Separate the z-scored PSDs back into stimulation ON and OFF lists
        psd_stim_list = [combined_psd_list[i] for i in range(len(combined_psd_list)) if labels[i] == 'stim']
        psd_non_stim_list = [combined_psd_list[i] for i in range(len(combined_psd_list)) if labels[i] == 'non_stim']

        # Compute the mean PSD for stimulation OFF
        mean_psd_non_stim = np.mean(psd_non_stim_list, axis=0)

        # Apply baseline correction to the stimulation ON PSDs
        psd_stim_corrected = baseline_correct_psd_on(psd_stim_list, mean_psd_non_stim)
        psd_non_stim_corrected = baseline_correct_psd_off(psd_non_stim_list, mean_psd_non_stim)

        # Plot average PSD for stimulation ON and OFF in the same plot
        plot_avg_psd(psd_stim_list, psd_non_stim_list, fmax, f'Average PSD {file_path} without correction', raw.info['ch_names'])
        plot_avg_psd(psd_stim_corrected, psd_non_stim_corrected, fmax, f'Average PSD {file_path} with correction', raw.info['ch_names'])

        # Plot all epochs in one big subplot figure
        # plot_all_epochs_avg(psd_stim_corrected, psd_non_stim_corrected, freqs, f'All Epochs PSD {file_path}', raw.info['ch_names'])

        # Plot time series for a single epoch
        if subepochs_stim:
           # Prompt the user to enter the subepoch index
            subepoch_index = int(input(f"Enter the subepoch index (0 to {len(subepochs_non_stim) - 1}): "))
            # Validate the input index
            if subepoch_index < 0 or subepoch_index >= len(subepochs_stim):
                print("Invalid index. Please enter a number between 0 and", len(subepochs_stim) - 1)
            else:
                # Call the plotting function with the user-defined index
                plot_single_epoch_time_series(subepochs_stim, subepoch_index=subepoch_index, title='Time Series for Subepoch (Stimulation ON)')
        if subepochs_non_stim:
            # Prompt to enter the subepoch index
            subepoch_index = int(input(f"Enter the subepoch index (0 to {len(subepochs_non_stim) - 1}): "))
            # Validate the input index
            if subepoch_index < 0 or subepoch_index >= len(subepochs_non_stim):
                print("Invalid index. Please enter a number between 0 and", len(subepochs_non_stim) - 1)
            else:
                # Call the plotting function with the user-defined index
                plot_single_epoch_time_series(subepochs_non_stim, subepoch_index=subepoch_index, title='Time Series for Subepoch (Stimulation OFF)')

        # Plot PSD for a single channel and epoch before and after baseline correction
        # plot_single_epoch_psd(subepochs_stim[-1], subepochs_non_stim[-1], mean_psd_non_stim, sfreq, fmax, 0, 'PSD of Subepoch')

if __name__ == "__main__":
    main()
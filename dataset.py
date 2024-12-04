import numpy as np
import pandas as pd
import os 
import math
import mne
import mne_connectivity
import torch
from torch_geometric.data import Data, Dataset


class MEGGraphs(Dataset):
    '''
    Creates a Dataset object of graphs out of the MEG data. 
    '''
    def __init__(self, root, filenames, stim_info_dict, duration, overlap, conn_method, freqs, fmax, ramp_time):
        '''
        Initializes all the inputs given to the class. 
        INPUTS: 
            - root              : relative path where folders 'raw' and 'processed' are located
            - filenames         : list of filenames of raw data
            - stim_info_dict    : dictionary with stimulation information per file
            - duration          : duration of final epochs in seconds
            - overlap           : overlap between final epochs in seconds
            - conn_method       : string of connectivity metric
            - freqs             : dictionary of frequencies for connectivity calculation
            - fmax              : maximum frequency for PSD calculation
            - ramp_time         : time in seconds that is removed in the beginning and end of each epoch

        OUTPUT: N/A
        '''
        # Initialize inputs
        self.filenames = filenames
        self.stim_info_dict = stim_info_dict
        self.duration = duration
        self.overlap = overlap 
        self.conn_method = conn_method
        self.freqs = freqs
        self.fmax = fmax
        self.ramp_time = ramp_time

        # Initialize lists to hold epoch counts and graphs, will be calculated and filled later
        self.amount_epochs = None
        self.graphs = []

        # Retrieve the basic functionality from torch_geometric.data.Dataset
        super().__init__(root)

    @property
    def raw_file_names(self):
        '''
        If this file already exists in raw_dir, 'def download' is skipped. Since 'def download' is passed, make sure the raw (or preprocessed) data file does exist in raw_dir.
        INPUT: N/A
        OUTPUT: N/A
        '''
        return self.filenames

    @property
    def processed_file_names(self):
        '''
        If these files already exist in processed_dir, 'def process' is skipped. 
        If you have made changes to the processing and want to test those without having to delete the already existing graph files, you can comment the return statement and return a random string instead.
        INPUT: N/A
        OUTPUT: N/A
        '''
        # Define the amount of epochs (i.e. graphs) to be generated
        if self.amount_epochs is None:
            self.amount_epochs, self.epochs_per_file = self.compute_amount_epochs()

        # Define the names of saved graphs
        file_names = []
        for idx_files in range(len(self.filenames)):
            for idx_epoch in range((self.epochs_per_file[idx_files])):
                file_names.append(f'graph_{idx_files}_{idx_epoch}.pt')
        
        # Check if the files already exist in processed_dir
        all_files_exist = True
        for file_name in file_names:
            full_path = os.path.join(self.processed_dir, file_name)
            if not os.path.exists(full_path):
                all_files_exist = False
                break
        
        # If all files already exist, reload the graphs to the list 'self.graphs'
        if all_files_exist:
            print("All files already exist, reloading graphs to graphs list...")
            self.load_existing_graphs()
        # If the files do not exist, generate the graphs using 'process()' and save them to the list 'self.graphs'
        else:
            print(f"A total of {self.amount_epochs} files will be generated")

        return file_names
        # return 'random'

    @property
    def raw_paths(self):
        '''
        Defines the paths where the (raw) data can be found.
        INPUT: N/A
        OUTPUT: N/A
        '''
        raw_paths = [os.path.join(self.raw_dir, filename) for filename in self.filenames]
        return raw_paths
    
    def download(self):
        '''
        Downloads data. 
        Since our data is already downloaded, this function is passed.
        INPUT: N/A
        OUTPUT: N/A
        '''
        pass

    def compute_amount_epochs(self):
        '''
        Downloads data. 
        Since our data is already downloaded, this function is passed.
        INPUT: N/A
        OUTPUT:
            - epochs_per_file   : List of epochs for each file
            - total_epochs      : Total amount of epochs across all files
        '''
        total_epochs = 0  # Initialize the total number of epochs
        epochs_per_file = []  # List to store the number of epochs for each file

        for filename in self.filenames:
            raw_path = os.path.join(self.raw_dir, filename)  # Assuming you have a raw_dir attribute
            raw = self.load_raw_data(raw_path)  # Load raw data to get n_times
            time_points = raw.n_times  # Get the number of time points inside the raw object
            sfreq = raw.info['sfreq']  # Get the sampling frequency
            total_duration = time_points / sfreq  # Total duration of the file in seconds

            # Retrieve stimulation info for the current file
            stim_info = self.stim_info_dict.get(filename, None)
            if stim_info is None:
                raise ValueError(f"No stimulation information found for {filename}")

            # Extract OFF times
            first_off_time = float(stim_info['First_OFF_time'])
            last_off_time = float(stim_info['Last_OFF_time'])

            # Initialize the number of epochs
            num_epochs = 0

            # Check if the first OFF time is greater than 60 seconds (i.e. file starts with stimulation ON event)
            if first_off_time > 60:
                # Calculate the number of full 60-second (stimulation ON) epochs before the first OFF time
                num_epochs += int(first_off_time // 60)

            # Check if the last OFF time is close to the end of the file to check if an epoch can be present after the last OFF time
            # if last_off_time <= total_duration - 5:  # Assuming "close" means within 5 seconds of the end
                # Calculate the number of full 60-second epochs from the last OFF time to the end of the file
                # num_epochs += int((total_duration - last_off_time) // 60)

            # Calculate the number of 60-second epochs that can fit in the remaining time (in between last and first OFF times)
            remaining_time = last_off_time - first_off_time
            if remaining_time > 0:
                # Calculate the number of additional epochs based on overlap and add to 'num_epochs'
                additional_epochs = int(remaining_time // 60)
                num_epochs += additional_epochs
            
            # Calculate total amount of subepochs based on duration and overlap
            length_epoch = 60 - 2 * self.ramp_time  # Specify the length of one epoch
            if self.overlap == 0:
                epochs_for_file = int((length_epoch - self.duration) / self.duration) * num_epochs
            else:
                epochs_for_file = int((length_epoch - self.duration) / (self.duration - self.overlap) + 1) * num_epochs
            
            epochs_per_file.append(epochs_for_file)  # Append the number of epochs for this file to list
            total_epochs += epochs_for_file  # Update the total number of epochs

        print(f'Total amount of epochs across all files: {total_epochs}')
        print(f'Epochs per file: {epochs_per_file}')
        return total_epochs, epochs_per_file
    
    def load_existing_graphs(self):
        '''
        Load existing graphs from the processed directory into the list 'self.graphs'.
        INPUT: N/A
        OUTPUT: N/A
        '''
        for idx_files in range(len(self.filenames)):
            for idx_epoch in range(self.epochs_per_file[idx_files]):
                file_name = f'graph_{idx_files}_{idx_epoch}.pt'
                full_path = os.path.join(self.processed_dir, file_name)
                graph = torch.load(full_path, weights_only=False)
                self.graphs.append(graph)

    def process(self):
        '''
        Performs all the processing steps needed to turn the raw (or preprocessed) data into graphs.
        INPUT: N/A
        OUTPUT: N/A
        '''
        # Initialize the self.graphs list 
        self.graphs = []

        # Iterate over each of the (raw) data files
        for idx_files, filename in enumerate(self.filenames):
            print(f'Processing file {filename}')

            # Update current filename
            self.filename = filename  
            raw_path = self.raw_paths[idx_files]
      
            # Load data and keep relevant channels
            self.raw = self.load_raw_data(raw_path)

            # Retrieve the stimulation info for the current filename
            stim_info = self.stim_info_dict.get(filename, None)
            if stim_info is None:
                raise ValueError(f"No stimulation information found for {filename}")
           
            # Define events stating when stimulation did or did not take place
            self.events, self.event_id = self.create_events(stim_info)

            # Create epochs during which stimulation took place
            self.epochs_stim = self.create_epochs(self.raw,
                                                self.events, 
                                                self.event_id,
                                                self.ramp_time, 
                                                label='stim'
                                                )    

            # Create epochs during which no stimulation took place
            self.epochs_non_stim = self.create_epochs(self.raw,
                                                    self.events, 
                                                    self.event_id,
                                                    self.ramp_time,
                                                    label='non_stim'
                                                    )
            
            # Check the number of epochs created
            print(f"Stim epochs: {len(self.epochs_stim)}, Non-stim epochs: {len(self.epochs_non_stim)}")

            # Split epochs into lists of subepochs with the initialized duration and overlap
            self.subepochs_stim = self.split_epochs(self.epochs_stim, self.duration, self.overlap, self.ramp_time, label='stim')
            self.subepochs_non_stim = self.split_epochs(self.epochs_non_stim, self.duration, self.overlap, self.ramp_time, label='non_stim')

            # Check the number of subepochs created
            print(f"Subepochs stim: {len(self.subepochs_stim)}, Subepochs non-stim: {len(self.subepochs_non_stim)}")

            # Concatenate lists of subepochs together into an EpochsArray object
            self.subepochs = self.concatenate_subepochs(self.subepochs_stim, self.subepochs_non_stim)

            # Create a graph for each subepoch
            for idx_epoch in range(len(self.subepochs)):
                # Specify graph index for saving
                # idx_save = idx_epochs
                print(f'Processing graph_{idx_files}_{idx_epoch}')

                # Load data of current epoch
                epoch = self.subepochs.load_data()[idx_epoch]

                # Define correct label 
                if self.subepochs.events[idx_epoch, 2] == self.event_id['stim']:
                    label = 'stim' 
                elif self.subepochs.events[idx_epoch, 2] == self.event_id['non_stim']:
                    label = 'non_stim'

                # Resample for shorter runtime 
                resample_freq = 256
                epoch_resampled = epoch.resample(sfreq=resample_freq)

                # Get nodes with features
                nodes = self._get_nodes(epoch_resampled, resample_freq, self.fmax)

                # Get edges with weights
                edge_index, edge_weight = self._get_edges(epoch_resampled,      
                                                        resample_freq, 
                                                        self.conn_method,
                                                        self.freqs)

                # Define label
                y = self._get_labels(label)

                # Create graph
                graph = Data(x=nodes, edge_index=edge_index, edge_attr=edge_weight, y=y)

                # Save graph with correct index
                self.graphs.append(graph)  # Store the graph to the graph list
                torch.save(graph, os.path.join(self.processed_dir, f'graph_{idx_files}_{idx_epoch}.pt'))

    def load_raw_data(self, file_path):
        '''
        Loads the MEG data as fif-file and picks the selected channels from each brain region. 
        INPUT: 
            - file_path      : path to the data file currently being processed
        
        OUTPUT:
            - raw            : raw object of time series data of selected channels
        '''
        # Load raw data file
        raw = mne.io.read_raw_fif(file_path, preload=False)

        # Define which channels you want to keep
        central_channels = ['MLO31', 'MLO33', 'MLO34',
                            'MZO02',
                            'MRO31', 'MRO33', 'MRO34',
                            'MLT32', 'MLT33', 'MLT36', 'MLT24',
                            'MLF22', 'MLF43', 'MLF55',
                            'MZF02', 'MZF03', 
                            'MRF22', 'MRF43', 'MRF55',
                            'MRT32', 'MRT33', 'MRT36', 'MRT24',
                            'MRC21', 'MRC23', 'MRC53', 'MRC32',
                            'MLC21', 'MLC23', 'MLC53', 'MLC32',
                            'MLP33', 'MLP44', 'MLP41', 'MLP54', 'MLP57',
                            'MRP33', 'MRP44', 'MRP41', 'MRP54', 'MRP57',
                            'MZP01',
                            'MZC01', 'MZC02', 'MZC03','MZC04']
        
        # Pick only the time series of the defined channels 
        raw.pick_channels(central_channels)
        return raw      
    
    def create_events(self, stim_info):
        '''
        Defines the events needed to create epochs. Events need three columns: 
        (1) the sample of the event onset;
        (2) all zeros (in most cases);
        (3) the event_id labelling the type of event.

        Events_times represents the times at which the stimulation was switched off.

        INPUT:
            - stim_info     : dictionary with information about the stimulation times
        OUTPUTS:
            - events        : array with events information needed for epoch creation
            - event_id      : dictionary with labels for stimulation ON and OFF

        '''
        # Define the first time point where stimulation was turned OFF (in seconds)
        first_stim_off_time = float(stim_info['First_OFF_time'])
        second_stim_off_time = float(stim_info['Second_OFF_time'])
        last_stim_off_time = float(stim_info['Last_OFF_time'])

        # Define the duration of the ON and OFF cycles
        cycle_duration = abs(second_stim_off_time - first_stim_off_time)  # number of seconds OFF and ON in total
        print('cycle duration', cycle_duration)
        # Define number of cycles in the recording
        comp_num_cycles = (last_stim_off_time - first_stim_off_time) / cycle_duration
        threshold = 0.9 # Define a threshold for rounding up
        num_cycles = math.ceil(comp_num_cycles) if (comp_num_cycles % 1) > threshold else int(comp_num_cycles)
        print('number of cycles:', num_cycles)

        # Define sampling frequency of raw data 
        sfreq = 2400

        # Calculate the time points for stimulation OFF events
        events_off_times = []
        
        # Add the first OFF and ON time
        events_off_times.append(first_stim_off_time)

        # Calculate subsequent stimulation OFF times
        for i in range(1, num_cycles):
            # Calculate OFF time for each cycle 
            off_time = first_stim_off_time + i * cycle_duration
            events_off_times.append(off_time)
        
        # Check if the last stimulation OFF time is already created
        if last_stim_off_time in events_off_times:
            events_off_times = events_off_times
        else: 
            # Add the last stimulation OFF time if not created yet
            events_off_times.append(last_stim_off_time)

        # Convert the time points to samples
        events_off_samples = [int(time * sfreq) for time in events_off_times]
        print('event off times', events_off_times)

        # Generate event IDs
        event_id = {'stim': 1, 'non_stim': 0}

        # Define number of events (normally 'num_cycles'+1)
        num_events = len(events_off_samples)

        # Define the start time as empty if the file does not start with a stimulation ON event
        event_start_time = []

        # Check if the first_stim_off_time is larger than 60 seconds, if so, file starts with stimulation ON event
        if first_stim_off_time > 60:
            # Compute the event start time
            event_start_time = first_stim_off_time - (cycle_duration / 2)
            # If file starts with stimulation ON event, add one to the number of events
            num_events = num_events + 1

        print('num events', num_events)

        # Create an array of zeroes with 2 * (num_events - 1) rows and 3 columns 
        events = np.zeros((2 * (num_events - 1), 3))

        # If file starts with stimulation ON event, add this to the events
        if event_start_time:
            # Add first_stim_off_time to the events with label 'stim'
            events[0] = [events_off_samples[0], 0, event_id['stim']]
            event_index_offset = 1  # Start filling consequent events from the second row
        else:
            event_index_offset = 0 # Fill events from the first row (file starts with stimulation OFF event)

        # Fill first (num_events - 1) rows with event samples and labels for stimulation ON
        for idx, event in enumerate(events_off_samples[1:]):
            events[idx + event_index_offset] = [event, 0, event_id['stim']]

        # Fill last (num_events - 1) rows with event samples and labels for stimulation OFF 
        for idx, event in enumerate(events_off_samples[:-1]):
            events[idx + (num_events - 1)] = [event, 0, event_id['non_stim']]

        # Turn all values in events array into integers 
        events = events.astype(int)

        # Keep only events with non-zero sample
        events = events[events[:, 0] != 0]
        return events, event_id

    def create_epochs(self, raw, events, event_id, ramp_time, label):
        '''
        Creates 60-second epochs based on the events defined in 'create_events'. 
        Stimulation was turned ON in the 60 seconds prior to the event sample; stimulation was turned OFF in the 60 seconds following the event sample. 
        INPUTS:
            - raw           : raw object of time series data of central channels
            - events        : array with events information needed for epoch creation
            - event_id      : dictionary with labels for stimulation ON and OFF
            - ramp_time     : time in seconds that is removed in the beginning and end of each epoch (to account for ramping of stimulation effects)
            - label         : string defining whether this epoch is for stimulation ON or OFF

        OUTPUT:
            - epochs        : Epochs object containing five 60-second stimulation ON epochs, and five stimulation OFF epochs
        '''
        # Define tmin and tmax of the epoch you want to create relative to the event sample
        if label == 'stim': 
            tmin, tmax = -60 + ramp_time, -ramp_time
        elif label == 'non_stim':
            tmin, tmax = ramp_time, 60 - ramp_time

        print(f"Creating epochs for {label}: tmin={tmin}, tmax={tmax}")
        print(f"Raw data duration: {raw.times[-1]} seconds")
        print(f"Events: {events}")

        # Create epochs
        epochs = mne.Epochs(raw,
                            events,
                            event_id=event_id[label],
                            tmin=tmin,
                            tmax=tmax,
                            preload=False,
                            baseline=None)
        epochs.drop_bad()
        print(f"Number of epochs created: {len(epochs)}")
        return epochs

    def split_epochs(self, epochs, duration, overlap, ramp_time, label):
        '''
        Splits each epoch into subepochs of initialized duration and overlap and returns a list of all subepochs. 
        INPUTS:
            - epochs            : Epochs object containing five 60-second stimulation ON epochs, and five stimulation OFF epochs
            - duration          : duration of final epochs in seconds
            - overlap           : overlap between final epochs in seconds
            - ramp_time         : time in seconds that is removed in the beginning and end of each epoch (to account for ramping of stimulation effects)
            - label             : string defining whether this epoch is for stimulation ON or OFF

        OUTPUT: 
            - subepochs_list    : list of subepochs of length 'duration'

        NB: since an epochs object cannot handle multiple epochs with the exact same event sample, the event samples of the subepochs are slightly altered using 'unique_event_samples'. 
        '''
        # Define epoch length
        length_epoch = 60 - 2 * ramp_time

        # Duration is not corresponding to the epoch length, so cropping is needed
        if duration != length_epoch:
            # Since tmin and tmax are different for stimulation ON and OFF epochs, these need to be split    
            if label == 'stim':
                # First subepoch needs to start at -60 seconds, excluding the ramp time
                start = -60 + ramp_time  # This is the earliest start time

                # Last subepoch needs to start at 'duration' before the ramp time at the end of the epoch
                stop = -ramp_time - duration  # This is the latest start time

                # Ensure that the last tmin does not exceed the minimum required value
                max_last_tmin = -ramp_time - duration
                if stop < max_last_tmin:
                    stop = max_last_tmin
                
                # Calculate how many subepochs you will get out of the 60 seconds based on 'duration' and 'overlap' 
                num = int((length_epoch - duration) / (duration - overlap) + 1)

                # Create list of all start time points of subepochs
                all_tmin = np.linspace(start, stop, num)
                print('all_tmin', all_tmin)

                # Create list of all end time points of subepochs 
                all_tmax = all_tmin + duration
                print('all_tmax', all_tmax)

                # All subepochs will have the same event sample, but is not possible in an Epochs object.
                # Therefore, create a list of unique (even) numbers to make each event sample unique. 
                unique_event_samples = list(range(0, 2*num, 2))

            elif label == 'non_stim':
                # First subepoch needs to start at -60 seconds 
                start = ramp_time

                # Last subepoch needs to start at 'duration' before 60
                stop = int(60 - ramp_time - duration)

                # Calculate how many subepochs you will get out of the 60 seconds based on 'duration' and 'overlap' 
                num = int((length_epoch - duration) / (duration - overlap) + 1)

                # Create list of all start time points of subepochs
                all_tmin = np.linspace(start, stop, num)
                print('all_tmin', all_tmin)

                # Create list of all end time points of subepochs 
                all_tmax = all_tmin + duration
                print('all_tmax', all_tmax)

                # Create list of OTHER unique (odd) numbers than for stimulation ON
                unique_event_samples = list(range(1, 2*num, 2))
            
            # Define empty list to fill with subepochs
            subepochs_list = []

            # Iterate over all epochs
            for idx, _ in enumerate(range(epochs.__len__())):
                # Iterate over all tmin and tmax
                for i, (tmin, tmax) in enumerate(zip(all_tmin, all_tmax)):
                    # Load data from epoch
                    epoch = epochs.load_data()[idx]

                    # Crop epoch with tmin and tmax
                    subepoch = epoch.crop(tmin=tmin, tmax=tmax)

                    # Create unique event sample
                    subepoch.events[:, 0] = subepoch.events[:, 0] + unique_event_samples[i]

                    # Add subepoch to list
                    subepochs_list.append(subepoch)

        # Duration is corresponding to the epoch length, so no cropping is needed
        else:       
            # Define empty list to fill with subepochs
            subepochs_list = []
            
            # Iterate over all epochs
            for idx, _ in enumerate(range(epochs.__len__())):
                # Load data from epoch
                epoch = epochs.load_data()[idx]

                # Make sure stimulation OFF epochs have a slightly different event sample than stimulation ON epochs
                if label == 'non_stim':
                    epoch.events[:, 0] = epoch.events[:, 0] + 1 
                
                # Add subepoch to list
                subepochs_list.append(epoch)

        print('subepoch', subepochs_list)
        return subepochs_list
    
    def concatenate_subepochs(self, subepochs_list_stim, subepochs_list_non_stim):
        '''
        Concatenates all subepochs in the subepochs lists into one EpochsArray object. 
        INPUTS:
            - subepochs_list_stim       : list of subepochs of length 'duration' of stimulation ON
            - subepochs_list_non_stim   : list of subepochs of length 'duration' of stimulation OFF 

        OUTPUT:
            - combined_epochs           : EpochsArray object of all subepochs
        '''
        # Extract the data and events from the list of epochs
        all_data_stim = [epochs.get_data() for epochs in subepochs_list_stim]
        all_events_stim = [epochs.events for epochs in subepochs_list_stim]  

        all_data_non_stim = [epochs.get_data() for epochs in subepochs_list_non_stim] 
        all_events_non_stim = [epochs.events for epochs in subepochs_list_non_stim]

        # Concatenate stim and non_stim data and events
        all_data = np.concatenate((all_data_stim, all_data_non_stim), axis=0)
        all_events = np.concatenate((all_events_stim, all_events_non_stim), axis=0)

        combined_data = np.concatenate(all_data, axis=0)
        combined_events = np.concatenate(all_events, axis=0)

        # Use the info from one of the original epochs
        info = subepochs_list_stim[0].info

        # Create the combined Epochs object
        combined_epochs = mne.EpochsArray(combined_data, 
                                          info, 
                                          events=combined_events)

        return combined_epochs

    def _get_nodes(self, epoch, sfreq, fmax):
        '''
        Calculates the Power Spectral Density (PSD) for each of the central channels. This PSD can then be used as a node feature matrix, with each central channel as a node and its PSD as the node features. 
        INPUTS: 
            - epoch         : the epoch currently being processed
            - sfreq         : the defined resampling frequency
            - fmax          : the defined maximum frequency for the PSD calculation
        
        OUTPUT:
            - nodes         : Torch tensor object of the node feature matrix 
        '''
        # Perform PSD calculation
        psd, _ = mne.time_frequency.psd_array_welch(epoch.get_data()*10**15,
                                                    fmax=fmax, 
                                                    sfreq=sfreq,
                                                    n_fft=sfreq * 2)
        
        # Print the shape of the PSD and frequency bins
        print('PSD shape:', psd.shape)

        # Turn PSD into a torch tensor to get the node feature matrix 
        nodes = torch.tensor(np.squeeze(psd), dtype=torch.float)
        return nodes
    
    def _get_edges(self, epoch, sfreq, method, freqs): 
        '''
        Calculates a connectivity metric between each of the nodes, based on the method you provide as an input. 
        Based on the non-zero indices of the resulting connectivity matrix, the edges are defined. The actual values of the resulting connectivity matrix represent the edge weights. 
        INPUTS:
            - epoch         : the epoch currently being processed
            - sfreq         : the defined resampling frequency
            - method        : string of connectivity metric 
            - freqs         : dictionary of frequencies for connectivity calculation
        
        OUTPUT:
            - edge_index    : Torch tensor object of indices of connected nodes (edges)
            - edge_weight   : Torch tensor object of PLI-values (edge features)

        ''' 

        # Perform connectivity calculation
        conn = mne_connectivity.spectral_connectivity_time(
            epoch,
            freqs=freqs['freqs'],
            method=method,
            sfreq=sfreq,
            fmin=freqs['fmin'],
            fmax=freqs['fmax'],
            faverage=True,
        )

        # Get data as connectivity matrix
        conn_data = conn.get_data(output="dense")
        conn_data = np.squeeze(conn_data.mean(axis=-1))

        # Retrieve all non-zero elements from PLI with in each column the start and end node of the edge
        edges = np.array(np.nonzero(conn_data))

        # Convert edges to tensor
        edge_index = torch.tensor(edges, dtype=torch.long)

        # Retrieve the value of the edges from PLI
        edge_weight = torch.tensor(conn_data[edges[0], edges[1]], dtype=torch.float)

        return edge_index, edge_weight
    
    def _get_labels(self, label):
        '''
        Defines the label of the graph: stimulation ON or OFF.
        INPUT:
            - label         : string defining whether this epoch is for stimulation ON or OFF
        
        OUTPUT:
            - label_tensor  : Torch tensor object of the label (0 for OFF; 1 for ON)
        '''
        label = np.asarray([self.event_id[label]])
        label_tensor = torch.tensor(label, dtype=torch.int64)
        return label_tensor

    def len(self):
        '''
        Returns the number of data objects stored in the dataset. 
        INPUT: N/A
        OUTPUT:
            - total             : integer representng the amount of graphs in the dataset
        '''
        total = self.amount_epochs
        return total
    
    def graphs_per_file(self):
        '''
        Returns the number of data objects created per file.
        INPUT: N/A
        OUTPUT:
            - total_per_file    : list representing the amount of graphs per file
        '''
        total_per_file = self.epochs_per_file
        return total_per_file

    def get(self, file_idx, graph_idx):
        '''
        Gets the data object at index 'idx'
        This is equivalent to __getitem__ in pytorch
        INPUT:
            - file_idx      : integer defining from which file the graph should be retrieved
            - graph_idx     : integer defining which graph you want to retrieve
        
        OUTPUT:
            - graph         : Data object of graph number 'idx'
        '''
        graph = torch.load(os.path.join(self.processed_dir, f'graph_{file_idx}_{graph_idx}.pt'), weights_only=False)
        return graph

    def __getitem__(self, idx):
        '''
        Retrieve a graph from the dataset based on the index.
        INPUT:
            - idx           : integer index for the graph
        OUTPUT:
            - self.graphs   : the graph data corresponding to the index
        '''
        return self.graphs[idx]

import numpy as np
import pandas as pd
import os 
import math
import mne
import mne_connectivity
import torch
import glob
from torch_geometric.data import Data, Dataset
from concurrent.futures import ProcessPoolExecutor
import h5py
import copy
import gc
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt

class MEGGraphs(Dataset):
    '''
    Creates a Dataset object of graphs out of the MEG data. 
    '''
    def __init__(self, input_type, root, filenames, stim_info_dict, duration, overlap, conn_method, freqs, fmin, fmax, ramp_time, scout_data_list=None):
        '''
        Initializes all the inputs given to the class.

        INPUTS: 
            - input_type        : string defining the type of input data ('fif' or 'scouts')
            - root              : relative path where folders 'raw' and 'processed' are located
            - filenames         : list of filenames of raw data
            - stim_info_dict    : dictionary with stimulation information per file
            - duration          : duration of final epochs in seconds
            - overlap           : overlap between final epochs in seconds
            - conn_method       : string of connectivity metric
            - freqs             : dictionary of frequencies for connectivity calculation
            - fmin              : minimum frequency for PSD calculation
            - fmax              : maximum frequency for PSD calculation
            - ramp_time         : time in seconds that is removed in the beginning and end of each epoch
            - scout_data_list   : list of dictionaries with scout data, only added if input_type is 'scouts'

        OUTPUT: N/A
        '''

        # Initialize inputs
        self.input_type = input_type
        self.filenames = filenames
        self.stim_info_dict = stim_info_dict
        self.duration = duration
        self.overlap = overlap 
        self.conn_method = conn_method
        self.freqs = freqs
        self.fmin = fmin
        self.fmax = fmax
        self.ramp_time = ramp_time
        self.scout_data_list = scout_data_list
        self.top_k = None
        self.threshold = None

        # Initialize lists, will be calculated and filled later
        self.amount_epochs = None
        self.graphs = []
        self.sfreq = None

        # Retrieve the basic functionality from torch_geometric.data.Dataset
        super().__init__(root)

    @property
    def raw_file_names(self):
        '''
        If this file already exists in raw_dir, 'def download' is skipped. Since 'def download' is passed, make sure the data file does exist in raw_dir.

        INPUT: N/A
        OUTPUT: N/A
        '''

        return self.filenames

    @property
    def processed_file_names(self):
        '''
        If these files already exist in processed_dir, 'def process' is skipped. 
        If you have made changes to the processing and want to test those without having to delete the already existing graph files,
        you can comment the return statement and return a random string instead.

        INPUT: N/A
        OUTPUT: N/A
        '''

        # Define the amount of epochs (i.e. graphs) to be generated
        if self.amount_epochs is None:
            self.amount_epochs, self.epochs_per_file, self.stim_on_per_file, self.stim_off_per_file, self.sfreq = self.compute_amount_epochs()

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
        Compute the amount of subepochs (and hence, the amount of graphs) that are created in total and per file.
        
        INPUT: N/A
        OUTPUT:
            - epochs_per_file   : List of subepochs for each file
            - total_epochs      : Total amount of subepochs across all files
            - stim_on_per_file  : List of subepochs with stimulation ON for each file
            - stim_off_per_file : List of subepochs with stimulation OFF for each file
        '''

        # Initialize the total number of subepochs
        total_epochs = 0

        # List to store the number of subepochs for each file 
        epochs_per_file = []

        # List to store the number of subepochs with stim ON and OFF for each file 
        stim_on_per_file = []
        stim_off_per_file = []

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
            second_off_time = float(stim_info['Second_OFF_time'])
            last_off_time = float(stim_info['Last_OFF_time'])

            # Define the duration of the ON and OFF cycles
            cycle_duration = abs(second_off_time - first_off_time)  # number of seconds OFF and ON in total

            # Define number of cycles in the recording
            comp_num_cycles = (last_off_time - first_off_time) / cycle_duration
            threshold = 0.9  # Define a threshold for rounding up
            num_cycles = math.ceil(comp_num_cycles) if (comp_num_cycles % 1) > threshold else int(comp_num_cycles)

            # Initialize the number of epochs and number of events (stim ON and OFF)
            num_epochs = 0
            num_stim_on_events = 0
            num_stim_off_events = 0

            # Each cycle consists of one ON and one OFF epoch
            num_stim_on_events += num_cycles
            num_stim_off_events += num_cycles

            # Check if the first OFF time is greater than 60 seconds (i.e. file starts with stimulation ON event)
            if first_off_time > 60:
                # Add an epoch if a full 60-second stimulation ON event occurs before the first OFF time
                num_epochs += 1
                num_stim_on_events += 1

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
                # epochs_for_file = int((length_epoch - self.duration) / self.duration) * num_epochs
                epochs_for_file = int(length_epoch / self.duration) * num_epochs
                num_stim_on_epochs = int(length_epoch / self.duration) * num_stim_on_events
                num_stim_off_epochs = int(length_epoch / self.duration) * num_stim_off_events
            else:
                epochs_for_file = int((length_epoch - self.duration) / (self.duration - self.overlap) + 1) * num_epochs
                num_stim_on_epochs = int((length_epoch - self.duration) / (self.duration - self.overlap) + 1) * num_stim_on_events
                num_stim_off_epochs = int((length_epoch - self.duration) / (self.duration - self.overlap) + 1) * num_stim_off_events
            
            # Calculate amount of subepochs with stimulation ON and amount of subepochs with stimulation OFF
            epochs_per_file.append(epochs_for_file)  # Append the number of subepochs for this file to list
            total_epochs += epochs_for_file  # Update the total number of subepochs
            stim_on_per_file.append(num_stim_on_epochs)  # Append number of subepochs for this file with stimulation ON
            stim_off_per_file.append(num_stim_off_epochs)  # Append number of subepochs for this file with stimulation OFF
            print(f'Number of epochs for file {filename}: {epochs_for_file}')
            
        print(f'Total amount of epochs across all files: {total_epochs}')
        print(f'Epochs per file: {epochs_per_file}')
        print(f'Stim ON epochs per file: {stim_on_per_file}')
        print(f'Stim OFF epochs per file: {stim_off_per_file}')
        return total_epochs, epochs_per_file, stim_on_per_file, stim_off_per_file, sfreq
    
    def load_existing_graphs(self):
        '''
        Load existing graphs from the processed directory into the list 'self.graphs'.

        INPUT: N/A
        OUTPUT: N/A
        '''

        # Initialize lists to store the actual number of graphs created per file
        self.actual_epochs_per_file = []
        self.actual_stim_on_per_file = []
        self.actual_stim_off_per_file = []

        # Load graphs from processed directory
        for idx_files in range(len(self.filenames)):
            num_graphs = 0
            num_stim_on = 0
            num_stim_off = 0

            # Loop over each subepoch in the file and load the exisiting graph file
            for idx_epoch in range(self.epochs_per_file[idx_files]):
                file_name = f'graph_{idx_files}_{idx_epoch}.pt'
                full_path = os.path.join(self.processed_dir, file_name)
                graph = torch.load(full_path, weights_only=False)

                # Count the number of total graphs, stim ON and stim OFF graphs
                if graph.x is not None:
                    num_graphs += 1
                    if graph.y is not None:
                        if graph.y.item() == 1:
                            num_stim_on += 1
                        elif graph.y.item() == 0:
                            num_stim_off += 1
                
                # Append loaded graphs and actual number of graphs per file
                self.graphs.append(graph)
                self.actual_epochs_per_file.append(num_graphs)
                self.actual_stim_on_per_file.append(num_stim_on)
                self.actual_stim_off_per_file.append(num_stim_off)
    
    def create_raw_from_scouts(self, scouts_data):
        '''
        Creates an mne.io.RawArray object from a dictionary of scout signals.

        INPUT:
            - scouts_data : Dictionary of scout signals (keys are scout names, values are signals)

        OUTPUT:
            - raw_array : RawArray object
        '''

        # Extract sampling frequency
        sfreq = self.sfreq

        # Extract scout names and signals
        scout_names = list(scouts_data.keys())
        scout_signals = [scouts_data[scout_name] for scout_name in scout_names]

        # Ensure all scout signals have the same length
        signal_length = len(scout_signals[0])
        for signal in scout_signals:
            assert len(signal) == signal_length, "All scout signals must have the same length"

        # Create Info object
        info = mne.create_info(ch_names=scout_names, sfreq=sfreq, ch_types=['mag'] * len(scout_names))

        # Create RawArray
        scout_signals_array = np.array(scout_signals)
        raw_array = mne.io.RawArray(scout_signals_array, info)

        return raw_array
    
    def process(self):
        '''
        Performs all the processing steps needed to turn the raw (or preprocessed) data into graphs.

        INPUT: N/A
        OUTPUT: N/A
        '''

        # Initialize lists to store the actual number of graphs created per file
        self.actual_epochs_per_file = []
        self.actual_stim_on_per_file = []
        self.actual_stim_off_per_file = []

        # Parallelize graph creation by calling the process_file() function and an iterable of filenames
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(executor.map(self.process_file, enumerate(self.filenames)))

            # Collect results and store graphs in self.graphs
            for result in results:
                # Save all graphs created from one file
                graphs_file = result

                # Filter out empty graphs for processing
                valid_graphs_file = [graph for graph in graphs_file if graph.x is not None]
                # print('graph index', [(i, graph) for i, graph in enumerate(graphs_file)])

                # Compute mean PSD for stimulation OFF epochs per file
                mean_PSD_off = self.compute_mean_psd_off(graphs_file)

                # Apply baseline correction to all graphs based on the mean PSD of stimulation OFF epochs
                corrected_graphs = self.apply_baseline_correction(graphs_file, mean_PSD_off)
                print('corrected graph index', [(i, graph) for i, graph in enumerate(corrected_graphs)])
                print('len (corrected_graphs)', len(corrected_graphs))

                # Store corrected graphs in self.graphs
                self.graphs.extend(corrected_graphs)
                print(f'Graphs for file: {len(graphs_file)}')

                # # Track the actual number of graphs created per file
                # self.actual_epochs_per_file.append(len(valid_graphs_file))
                # self.actual_stim_on_per_file.append(sum(1 for graph in valid_graphs_file if graph.y is not None and graph.y.item() == 1))
                # self.actual_stim_off_per_file.append(sum(1 for graph in valid_graphs_file if graph.y is not None and graph.y.item() == 0))

                # Count stimulation ON graphs created per file
                stim_on_count = sum(1 for graph in valid_graphs_file if graph.y is not None and graph.y.item() == 1)
                self.actual_stim_on_per_file.append(stim_on_count)
                
                # Count stimulation OFF graphs created per file
                stim_off_count = sum(1 for graph in valid_graphs_file if graph.y is not None and graph.y.item() == 0)
                self.actual_stim_off_per_file.append(stim_off_count)

                # Count the total number of graphs created per file
                self.actual_epochs_per_file.append(len(valid_graphs_file))

                # Release memory for large data structures that are no longer needed
                del graphs_file, mean_PSD_off, corrected_graphs, stim_on_count, stim_off_count, valid_graphs_file
                gc.collect()

        print('Lenght self.graphs', len(self.graphs))

        # Create a copy of self.graphs to avoid modifying the original list
        graphs_copy = self.graphs.copy()

        # Save graphs to processed directory
        for idx_files in range(len(self.filenames)):
            print(f'Saving graphs for file {self.filenames[idx_files]}:')
            for idx_epoch in range(self.epochs_per_file[idx_files]):
                if graphs_copy:
                    graph = graphs_copy.pop(0)
                    print('graph index', idx_epoch)
                    print('graph pop', graph)
                    saved_graph = torch.save(graph, os.path.join(self.processed_dir, f'graph_{idx_files}_{idx_epoch}.pt'))
                    print('graph after saving', saved_graph)
                else:
                    print(f"No more graphs to save for file {self.filenames[idx_files]} at epoch {idx_epoch}")
                    break

        # Release memory for large data structures that are no longer needed
        del graphs_copy
        gc.collect()

    def process_file(self, idx_filename):
        '''
        Process a single file to create graphs.

        INPUT:
            - idx_filename:     Tuple containing index and filename

        OUTPUT:
            - graphs:           List of graphs created from the file
        '''

        idx_files, filename = idx_filename
        print(f'Processing file {filename}')

        # Update current filename
        self.filename = filename
        raw_path = self.raw_paths[idx_files]

        # Check input type of the file
        if self.input_type == 'fif':
            # Load data and keep relevant channels
            self.raw = self.load_raw_data(raw_path)
        
        # If input type is 'scout', load the scout data from the dictionary
        elif self.input_type == 'scout':
            scouts_data_dict = self.scout_data_list[idx_files]
            for scout_name, scout_signal in scouts_data_dict.items():
                print(f"Scout: {scout_name}, Signal shape: {scout_signal.shape}")
            self.raw = self.create_raw_from_scouts(scouts_data_dict)

        # Retrieve the stimulation info for the current filename
        stim_info = self.stim_info_dict.get(filename, None)
        if stim_info is None:
            return []

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
        self.subepochs_stim, self.bad_subepochs_stim = self.split_epochs(self.epochs_stim,
                                                                         self.duration,
                                                                         self.overlap,
                                                                         self.ramp_time,
                                                                         label='stim',
                                                                         threshold=100)
        self.subepochs_non_stim, self.bad_subepochs_non_stim = self.split_epochs(self.epochs_non_stim, 
                                                                                 self.duration,
                                                                                 self.overlap,
                                                                                 self.ramp_time,
                                                                                 label='non_stim',
                                                                                 threshold=100)

        # Check the number of subepochs created
        print(f"Subepochs stim: {len(self.subepochs_stim)}, Subepochs non-stim: {len(self.subepochs_non_stim)}")
        print(f'Bad subepochs stim ON: {self.bad_subepochs_stim}')
        print(f'Bad subepochs stim OFF: {self.bad_subepochs_non_stim}')

        # Concatenate lists of subepochs together into an EpochsArray object
        self.subepochs = self.concatenate_subepochs(self.subepochs_stim, self.subepochs_non_stim)

        # Combine list of bad subepochs and adjust indices for bad subepochs with stim OFF
        offset = len(self.subepochs_stim)
        adjusted_bad_subepochs_non_stim = [(total_idx + offset, subepoch) for total_idx, subepoch in self.bad_subepochs_non_stim]
        combined_bad_subepochs = self.bad_subepochs_stim + adjusted_bad_subepochs_non_stim
        print(f'Combined bad subepochs: {combined_bad_subepochs}')

        # Initialize a list to store graphs created from this file
        graphs_list = []

        # Initialize a list to store indices of bad subepochs
        bad_subepochs_indices = []

        # Create a graph for each subepoch
        total_subepochs = len(self.subepochs)
        for idx_epoch in tqdm(range(total_subepochs), desc=f"Processing graphs for file {filename}\n"):
            print(f'\nProcessing graph_{idx_files}_{idx_epoch}')

            # Check if the current subepoch is a bad subepoch
            is_bad_subepoch = any(idx_epoch == total_idx for total_idx, _ in combined_bad_subepochs)
            if is_bad_subepoch:
                print('total idx', [total_idx for total_idx, _ in combined_bad_subepochs])
                bad_subepoch = next((subepoch for total_idx, subepoch in combined_bad_subepochs if total_idx == idx_epoch), None)
                if bad_subepoch is not None:
                    # For each bad subepoch, make a plot and save it to the bad subepoch folder
                    avg_epoch_data = np.mean(bad_subepoch.get_data()*10**15, axis=1).squeeze()
                    plt.figure(figsize=(10, 6))
                    plt.plot(bad_subepoch.times, avg_epoch_data.T)
                    plt.title(f'Bad Subepoch: graph_{idx_files}_{idx_epoch}')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Signal (fT)')
                    plt.savefig(os.path.join(self.processed_dir, 'bad_subepochs', f'bad_subepoch_{idx_files}_{idx_epoch}.png'))
                    plt.close()

                    # Create graph for bad subepoch with empty attributes
                    bad_graph = Data(x=None, edge_index=None, edge_attr=None, y=None)
                    graphs_list.append(bad_graph)
                    bad_subepochs_indices.append(idx_epoch)
                    print(f'Bad subepoch, graph_{idx_files}_{idx_epoch} is empty.')
                else:
                    print('Bad subepoch not found')
            else:
                print(f'Good subepoch, creating graph_{idx_files}_{idx_epoch}...')
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

                # Convert data to appropriate unit, based on the data input type
                if self.input_type == 'fif':
                    # Convert data to femtoTesla (fT)
                    epoch_data = epoch_resampled.get_data() * 10**15
                elif self.input_type == 'scout':
                    # Convert data to picoAmpere (pA-m)
                    epoch_data = epoch_resampled.get_data() * 10**12

                print('Epoch data shape:', epoch_data.shape)
                # Get nodes with features
                nodes = self._get_nodes(epoch_data, resample_freq, self.fmin, self.fmax)

                # Get edges with weights
                edge_index, edge_weight = self._get_edges(epoch_data,
                                                    resample_freq, 
                                                    self.conn_method,
                                                    self.freqs,
                                                    idx_filename,
                                                    idx_epoch,
                                                    save_dir=f'F:\MEG GNN\GNN\Data\Connectivity\Subepoch_{self.duration}sec_{self.overlap}_overlap_freq_{self.fmin}_{self.fmax}')

                # Define label
                y = self._get_labels(label)

                # Create graph
                graph = Data(x=nodes, edge_index=edge_index, edge_attr=edge_weight, y=y)
                print(f'graph {idx_files} {idx_epoch}: {graph}')
                if graph.x is None:
                    print(f'LET OP, graph {idx_filename} {idx_epoch} is None')

                # Store the graph to the graph list
                graphs_list.append(graph)

                # Release memory for large data structures that are no longer needed
                del epoch, epoch_resampled, epoch_data, nodes, edge_index, edge_weight, y, graph
                gc.collect()

        # Save bad subepochs indices to a file
        bad_subepochs_dir = os.path.join(self.processed_dir, 'bad_subepochs')
        os.makedirs(bad_subepochs_dir, exist_ok=True)
        with open(os.path.join(bad_subepochs_dir, f'bad_subepochs_{idx_files}.txt'), 'w') as f:
            for idx in bad_subepochs_indices:
                f.write(f"{idx}\n")

        return graphs_list

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
    
    def load_avg_raw_data(self, file_path):
        '''
        Loads the MEG data as fif-file and picks the selected channels that are averaged, and returns a raw object
        with 'channels' (averaged signals of 3 channels each).

        INPUT: 
            - file_path      : path to the data file currently being processed
        
        OUTPUT:
            - avg_raw        : raw object of time series data of averaged channels
        '''

        # Load raw data file
        raw = mne.io.read_raw_fif(file_path, preload=False)
        
        # Select MEG channels to average over
        selected_channels = ['MLO31', 'MLO41', 'MLO22',
                    'MRO31', 'MRO41', 'MRO22',

                    'MLO23', 'MLO43', 'MLO12',
                    'MRO23', 'MRO43', 'MRO12',

                    'MLO34', 'MLO34', 'MLO34',
                    'MRO34', 'MRO34', 'MRO34',

                    'MZO02', 'MLO21', 'MRO21',

                    'MLT32', 'MLT41', 'MLT22',
                    'MLT33', 'MLT43', 'MLT23',
                    'MLT36', 'MLT45', 'MLT46',
                    'MLT24', 'MLT34', 'MLT35',
                    'MRT32', 'MRT41', 'MRT22',
                    'MRT33', 'MRT43', 'MRT23',
                    'MRT36', 'MRT45', 'MRT46',
                    'MRT24', 'MRT34', 'MRT35',

                    'MLF22', 'MLF31', 'MLF42', 
                    'MLF43', 'MLF53', 'MLF62', 
                    'MLF55', 'MLF56', 'MLF65', 
                    'MRF22', 'MRF31', 'MRF42',
                    'MRF43', 'MRF53', 'MRF62', 
                    'MRF55', 'MRF56', 'MRF65', 

                    'MZF02', 'MLF41', 'MRF41', 
                    'MZF03', 'MLC11', 'MRC11', 

                    'MRC21', 'MRC22', 'MRC52', 
                    'MRC23', 'MRC24', 'MRC31',
                    'MRC53', 'MRC42', 'MRC54', 
                    'MLC21', 'MLC22', 'MLC52', 
                    'MLC23', 'MLC24', 'MLC31',
                    'MLC53', 'MLC42', 'MLC54', 

                    'MLP23', 'MLP35', 'MLP12',
                    'MLP33', 'MLP42', 'MLP22', 
                    'MLP44', 'MLP56', 'MLP55', 
                    'MLP41', 'MLP32', 'MLP31', 
                    'MLP54', 'MLO14', 'MLO13', 
                    'MLP57', 'MLT14', 'MLT15', 
                    'MRP23', 'MRP35', 'MRP12',
                    'MRP33', 'MRP42', 'MRP22', 
                    'MRP44', 'MRP56', 'MRP55', 
                    'MRP41', 'MRP32', 'MRP31', 
                    'MRP54', 'MRO14', 'MRO13', 
                    'MRP57', 'MRT14', 'MRT15', 

                    'MZP01', 'MLP21', 'MRP21',
                    'MZC01', 'MLC51', 'MRC51',
                    'MZC02', 'MLC61', 'MRC61',
                    'MZC03', 'MLC63', 'MRC63',
                    'MZC04', 'MLP11', 'MRP11']
            
        # Check which channels are available in the raw data
        available_channels = [ch for ch in selected_channels if ch in raw.ch_names]
        print('Number of channels:', len(available_channels))
        
        # Pick only the time series of the defined channels 
        raw.pick_channels(available_channels)

        # Average the signals of every three selected channels
        averaged_data = []
        new_channel_names = []
        for i in range(0, len(available_channels), 3):
            if i + 2 < len(available_channels):
                # Average the signals of three channels
                avg_signal = np.mean(raw.get_data(picks=available_channels[i:i+3]), axis=0)
                averaged_data.append(avg_signal)
                new_channel_names.append(available_channels[i])  # Use the name of the first channel as the new channel name
        print('Number of nodes (averaged channels):', len(new_channel_names))

        # Create a new raw object with the averaged signals
        info = mne.create_info(ch_names=new_channel_names, sfreq=raw.info['sfreq'], ch_types='mag')
        averaged_data = np.array(averaged_data)
        avg_raw = mne.io.RawArray(averaged_data, info)
        
        return avg_raw
    
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

        # Define number of cycles in the recording
        comp_num_cycles = (last_stim_off_time - first_stim_off_time) / cycle_duration
        threshold = 0.9  # Define a threshold for rounding up
        num_cycles = math.ceil(comp_num_cycles) if (comp_num_cycles % 1) > threshold else int(comp_num_cycles)

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
    
        # High-pass filter the data to remove drifts
        raw.load_data().filter(l_freq=0.1, h_freq=None)

        # Define tmin and tmax of the epoch you want to create relative to the event sample
        if label == 'stim': 
            tmin, tmax = -60 + ramp_time, -ramp_time
        elif label == 'non_stim':
            tmin, tmax = ramp_time, 60 - ramp_time

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

    def split_epochs(self, epochs, duration, overlap, ramp_time, label, threshold):
        '''
        Splits each epoch into subepochs of initialized duration and overlap and returns a list of all subepochs.

        INPUTS:
            - epochs            : Epochs object containing stimulation ON and OFF epochs
            - duration          : duration of subepochs in seconds
            - overlap           : overlap between subepochs in seconds
            - ramp_time         : time in seconds that is removed in the beginning and end of each subepoch (to account for ramping of stimulation effects)
            - label             : string defining whether this epoch is for stimulation ON or OFF
            - threshold         : threshold for the subepochs to be considered as bad subepochs and therby removed from further analysis

        OUTPUT: 
            - subepochs_list    : list of subepochs of length 'duration'
            - bad_subepochs_list: list of bad subepochs of length 'duration'

        NB: since an epochs object cannot handle multiple epochs with the exact same event sample, the event samples of the subepochs are slightly altered using 'unique_event_samples'. 
        '''

        # Define epoch length
        length_epoch = 60 - 2 * ramp_time

        # Define emty lists to fill with subepochs
        subepochs_list = []
        bad_subepochs_list = []
        total_subepochs = 0

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

                # Create list of all end time points of subepochs 
                all_tmax = all_tmin + duration

                # All subepochs will have the same event sample, but this is not possible in an Epochs object
                # Therefore, create a list of unique (even) numbers to make each event sample unique
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

                # Create list of all end time points of subepochs 
                all_tmax = all_tmin + duration

                # Create list of OTHER unique (odd) numbers to make each event sample unique
                unique_event_samples = list(range(1, 2*num, 2))

            # Iterate over all epochs
            for idx, _ in enumerate(range(epochs.__len__())):
                # Counter for subepochs within the current epoch
                subepochs_count = 0

                # Load data from epoch
                epoch_data = epochs[idx].get_data() * 10**15

                # Calculate the median and MAD for the entire epoch
                epoch_median = np.median(epoch_data)
                epoch_mad = np.median(np.abs(epoch_data - epoch_median))

                # Iterate over all tmin and tmax
                for i, (tmin, tmax) in enumerate(zip(all_tmin, all_tmax)):
                    # Load data from epoch
                    epoch = epochs.load_data()[idx]

                    # Crop epoch with tmin and tmax
                    subepoch = epoch.crop(tmin=tmin, tmax=tmax)

                    # Detect bad epochs based on epoch median, MAD and maximum deviation of subepoch
                    subepoch_median = np.median(subepoch.get_data() * 10**15)
                    max_deviation = np.max(np.abs(subepoch.get_data() * 10**15 - epoch_median))
                    deviation_score = max_deviation / epoch_mad if epoch_mad != 0 else np.inf
                    # deviation_score = (max_deviation - epoch_median) / epoch_mad
                    
                    # If the deviation score exceeds the threshold, mark the epoch as bad
                    if deviation_score > threshold:
                        # Note that this subepoch will be included in further graph creation steps
                        bad_subepochs_list.append((total_subepochs, subepoch))

                    # Create unique event sample
                    subepoch.events[:, 0] = subepoch.events[:, 0] + unique_event_samples[i]

                    # Add subepoch to list
                    subepochs_list.append(subepoch)
                    subepochs_count += 1
                    total_subepochs += 1

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
                subepoch_count += 1
                total_subepochs += 1

        return subepochs_list, bad_subepochs_list
    
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

    def _get_nodes(self, epoch_data, sfreq, fmin, fmax):
        '''
        Calculates the Power Spectral Density (PSD) for each of the selected channels.
        This PSD can then be used as a node feature matrix.

        INPUTS: 
            - epoch_data    : The epoch currently being processed
            - sfreq         : The defined resampling frequency
            - fmin          : The defined minimum frequency for the PSD calculation
            - fmax          : The defined maximum frequency for the PSD calculation
        
        OUTPUT:
            - nodes         : Torch tensor object of the node feature matrix 
        '''

        # For time series data, perform PSD calculation with time series data in [fT]
        if self.input_type == 'fif':
            psd, _ = mne.time_frequency.psd_array_welch(epoch_data,
                                                        fmin=fmin,
                                                        fmax=fmax, 
                                                        sfreq=sfreq,
                                                        n_fft=sfreq,
                                                        average='mean',
                                                        remove_dc=True,
                                                        output='power')
        # For scout data, perform PSD calculation with scout data in [pA]
        elif self.input_type == 'scout':
            psd, _ = mne.time_frequency.psd_array_welch(epoch_data,
                                            fmin=fmin,
                                            fmax=fmax, 
                                            sfreq=sfreq,
                                            n_fft=sfreq,
                                            average='mean',
                                            remove_dc=True,
                                            output='power')

        print('psd shape', psd.shape)

        # Take the square root of the PSD to transform to [fT/sqrt(Hz)]
        psd_sqrt = np.sqrt(psd)

        # Turn PSD into a torch tensor to get the node feature matrix 
        nodes = torch.tensor(np.squeeze(psd_sqrt), dtype=torch.float)
        return nodes
    
    def _get_edges(self, epoch_data, sfreq, method, freqs, idx_file, idx_epoch, save_dir=f'F:\MEG GNN\GNN\Data\Connectivity'): 
        '''
        Calculates a connectivity metric between each of the nodes, based on the method you provide as an input.
        Based on the non-zero indices of the resulting connectivity matrix, the edges are defined.
        The resulting connectivity matrix represent the edge weights.

        INPUTS:
            - epoch_data    : The epoch currently being processed
            - sfreq         : The defined resampling frequency
            - method        : String of connectivity metric 
            - freqs         : Dictionary of frequencies for connectivity calculation
            - idx_file      : String of file path to save the connectivity matrix
            - idx_epoch     : String of epoch index to save the connectivity matrix
            - save_dir      : String of directory to save the connectivity matrix
        
        OUTPUT:
            - edge_index    : Torch tensor object of indices of connected nodes (edges)
            - edge_weight   : Torch tensor object of connectivity metric values (edge features)

        ''' 

        print('Epoch shape', epoch_data.shape)
        # Extract the filename from the idx_file tuple
        file_index, filename = idx_file

        # Define a pattern to search for the edges file -- Uncomment to search for previously saved edges
        if self.input_type == 'scout':
            pattern = os.path.join(save_dir, f"edges_(*, '{filename}')_{idx_epoch}_{method}_scout.pt")
        else:
            pattern = os.path.join(save_dir, f"edges_(*, '{filename}')_{idx_epoch}_{method}.pt")
        print('Pattern:', pattern)

        # Search for files matching the pattern
        matching_files = glob.glob(pattern)
        print('Matching files:', matching_files)

        # Check if any matching files are found -- Uncomment to load previously saved edges
        if matching_files:
            edge_filename = matching_files[0]
            print(f"Loading edges from {edge_filename}")
            edges = torch.load(edge_filename)
            return edges['edge_index'], edges['edge_weight']

        # Perform connectivity calculation
        conn = mne_connectivity.spectral_connectivity_time(
            epoch_data,
            freqs=freqs['freqs'],
            method=method,
            sfreq=sfreq,
            fmin=freqs['fmin'],
            fmax=freqs['fmax'],
            faverage=True,
            verbose=False
        )

        # Get data as connectivity matrix
        conn_data = conn.get_data(output="dense")
        conn_data = np.squeeze(conn_data.mean(axis=-1))
        print('Number of edges', np.size(conn_data))

        # Retrieve all non-zero elements from the connectivity matrix with in each column the start and end node of the edge
        edges = np.array(np.nonzero(conn_data))

        # Convert edges to tensor
        edge_index = torch.tensor(edges, dtype=torch.long)

        # Retrieve the value of the edges from the connectivity matrix
        edge_weight = torch.tensor(conn_data[edges[0], edges[1]], dtype=torch.float)
        print("Mean weigth:", edge_weight.mean())
        print("Max weight:", edge_weight.max())
        print("Min weight:", edge_weight.min())

        # Save the edges to disk -- Uncomment to save the edges
        os.makedirs(save_dir, exist_ok=True)
        if self.input_type == 'scout':
            save_edge_filename = os.path.join(save_dir, f"edges_{idx_file}_{idx_epoch}_{method}_scout.pt")
        elif self.input_type == 'fif':
            save_edge_filename = os.path.join(save_dir, f"edges_{idx_file}_{idx_epoch}_{method}.pt")
        torch.save({'edge_index': edge_index, 'edge_weight': edge_weight}, save_edge_filename)
        print(f"Edges saved to {save_edge_filename}")

        return edge_index, edge_weight
    
    def compute_mean_psd_off(self, graphs_file):
        '''
        Computes the mean Power Spectral Density (PSD) across all stimulation OFF subepochs for one file.
        
        INPUT:
            - graphs_file   : List of graphs created from one file
        OUTPUT:
            - mean_psd_off  : Mean PSD values for stimulation OFF subepochs
        '''

        # Initialize a list to store PSD values for stimulation OFF subepochs
        psd_off_list = []

        # Filter out empty graphs
        valid_graphs = [graph for graph in graphs_file if graph.x is not None and graph.y is not None]

        # Iterate over all valid graphs and collect PSD values for stimulation OFF subepochs
        for graph in valid_graphs:
            if graph.y.item() == 0:  # Label 0 indicates stimulation OFF
                psd_off_list.append(graph.x.numpy())
        
        # Convert list to numpy array for easier manipulation
        psd_off_array = np.array(psd_off_list)

        # Compute the mean PSD across all stimulation OFF subepochs
        mean_psd_off = np.mean(psd_off_array, axis=0)

        return mean_psd_off

    def apply_baseline_correction(self, graphs_file, mean_psd_off):
        '''
        Applies baseline correction to both stim ON and OFF subepochs for one file using the mean PSD of the stimulation OFF subepochs.

        INPUT:
            - mean_psd_off      : Mean PSD values for stimulation OFF subepochs for one file
            - graphs_file       : List of graphs created from one file
        OUTPUT:
            - corrected_graphs  : List of graphs with baseline correction applied
        '''
    
        # Initialize a list to store corrected graphs
        corrected_graphs = []

        # Iterate over all graphs and apply baseline correction
        for graph in graphs_file:
            if graph.x is not None:
                psd = graph.x.numpy()
                corrected_psd = (psd - mean_psd_off) / mean_psd_off
                graph.x = torch.tensor(corrected_psd, dtype=torch.float)
            corrected_graphs.append(graph)
        
        return corrected_graphs

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
            - total     : integer representng the amount of graphs in the dataset
        '''

        # Filter out empty graphs
        non_empty_graphs = [graph for graph in self.graphs if graph.x is not None]

        return len(non_empty_graphs)
    
    def get_filenames(self):
        '''
        Returns the filenames of the files that were processed.

        INPUT: N/A
        OUTPUT:
            - filenames  : list of filenames of the files that were processed
        '''

        filenames = self.filenames
        return filenames

    def get_indices_by_label(self, label):
        '''
        Retrieves the indices of all graphs with the specified label.

        INPUT:
            - label :   Label of the graphs to retrieve indices for (e.g., 'stim' or 'non_stim')

        OUTPUT:
            - indices : List of indices of graphs with the specified label
        '''

        indices = []
        target_label = 1 if label == 'stim' else 0
        for idx, graph in enumerate(self.graphs):
            if graph.y.item() == target_label:
                indices.append(idx)
        return indices
    
    def graphs_per_file(self):
        '''
        Returns the number of data objects created per file.

        INPUT: N/A
        OUTPUT:
            - total_per_file    : list representing the amount of graphs per file
        '''

        total_per_file = self.actual_epochs_per_file
        return total_per_file
    
    def stim_graphs_per_file(self):
        '''
        Returns the number of graphs created with stimulation ON per file.

        INPUT: N/A
        OUTPUT:
            - stim_on_per_file      : list representing the amount of graphs for stim ON per file
        '''
        stim_on_per_file = self.actual_stim_on_per_file
        return stim_on_per_file
    
    def non_stim_graphs_per_file(self):
        '''
        Returns the number of graphs created with stimulation OFF per file.

        INPUT: N/A
        OUTPUT:
            - stim_off_per_file     : list representing the amount of graphs for stim OFF per file
        '''
        stim_off_per_file = self.actual_stim_off_per_file
        return stim_off_per_file

    def get(self, file_idx, graph_idx):
        '''
        Gets the data object at index 'idx'

        INPUT:
            - file_idx      : integer defining from which file the graph should be retrieved
            - graph_idx     : integer defining which graph you want to retrieve
        
        OUTPUT:
            - graph         : Data object of graph number 'idx'
        '''

        try:
            graph = torch.load(os.path.join(self.processed_dir, f'graph_{file_idx}_{graph_idx}.pt'), weights_only=False)
        except FileNotFoundError:
            print(f"Graph file graph_{file_idx}_{graph_idx}.pt not found.")
            return None
        return graph

    def __getitem__(self, idx):
        '''
        Pytorch specific function to retrieve a graph from the dataset based on the index.

        INPUT:
            - idx           : integer index for the graph
        OUTPUT:
            - self.graphs   : the graph data corresponding to the index
        '''

        # Filter out empty graphs
        non_empty_graphs = [graph for graph in self.graphs if graph.x is not None]

        return non_empty_graphs[idx]
    
    def remove_bad_graphs(self):
        '''
        Moves bad graph files to a separate folder for bad epoch and renames the remaining graph files in the processed directory.

        INPUT: N/A
        OUTPUT: N/A
        '''

        # Create a directory for bad subepochs if it doesn't exist
        bad_subepochs_dir = os.path.join(self.processed_dir, 'bad_subepochs')
        os.makedirs(bad_subepochs_dir, exist_ok=True)

        # Initialize a dictionary for the bad indices per file
        bad_indices_per_file = {}

        # For each file, retrieve the bad subepoch text file including the indices of the bad subepochs 
        for idx_files in range(len(self.filenames)):
            bad_subepochs_file = os.path.join(self.processed_dir, 'bad_subepochs', f'bad_subepochs_{idx_files}.txt')

            # Retrieve the indices of the bad subepochs for this file
            if os.path.exists(bad_subepochs_file):
                with open(bad_subepochs_file, 'r') as f:
                    bad_indices = [int(line.strip()) for line in f]
                    bad_indices_per_file[idx_files] = bad_indices
            else:
                print(f'The file "bad_subepochs_{idx_files}.txt" does not exist in this directory: {bad_subepochs_dir}!')

        # Move bad graphs for this file to the bad subepochs directory
        for idx_files in range(len(self.filenames)):
            for idx_epoch in range(self.epochs_per_file[idx_files]):
                if idx_epoch in bad_indices_per_file[idx_files]:
                    file_name = f'graph_{idx_files}_{idx_epoch}.pt'
                    full_path = os.path.join(self.processed_dir, file_name)
                    if os.path.exists(full_path):
                        shutil.move(full_path, os.path.join(bad_subepochs_dir, file_name))
                        print(f"Moved bad graph: {file_name}")

        # Reorder and rename the remaining graph files
        for idx_files in range(len(self.filenames)):
            # Specify new graph index to be 0
            new_idx = 0
            # Loop over all subepochs for one file and order the graph index, starting at 0
            for idx_epoch in range(self.epochs_per_file[idx_files]):
                file_name = f'graph_{idx_files}_{idx_epoch}.pt'
                full_path = os.path.join(self.processed_dir, file_name)
                if os.path.exists(full_path):
                    new_file_name = f'graph_{idx_files}_{new_idx}.pt'
                    new_full_path = os.path.join(self.processed_dir, new_file_name)
                    os.rename(full_path, new_full_path)
                    print(f"Renamed {file_name} to {new_file_name}")
                    new_idx += 1

import os
import numpy as np
import pandas as pd
import torch_geometric
import networkx as nx
import matplotlib.pyplot as plt
import mne


def plot_PSD_avg_per_file(dataset, fmax):
    '''
    Plots the average power spectral density (PSD) of graphs where stimulation was turned ON and where stimulation was turned OFF.
    
    INPUT:
        - dataset       : Dataset of graphs
        - fmax          : Maximum frequency to plot
    OUTPUT: N/A
    '''

    # Retrieve how many graphs there are in the dataset
    total_per_file = dataset.graphs_per_file()  # Get total number of graphs per file
    stim_on_per_file = dataset.stim_graphs_per_file()  # Get the number of stim ON epochs per file
    stim_off_per_file = dataset.non_stim_graphs_per_file()  # Get the number of stim OFF epochs per file

    # Loop through each file
    for file_index in range(len(total_per_file)):
        # Initialize accumulators for PSD for the current file
        psd_stim_file = []
        psd_non_stim_file = []

        # Loop through stimulation ON epochs
        for stim_index in range(stim_on_per_file[file_index]):
            data_stim = dataset.get(file_index, stim_index)
            psd_stim = data_stim.x.numpy()  # Retrieve the PSD
            psd_stim_file.append(psd_stim)

        # Loop through stimulation OFF epochs
        for non_stim_index in range(stim_off_per_file[file_index]):
            data_non_stim = dataset.get(file_index, total_per_file[file_index] - 1 - non_stim_index)
            psd_non_stim = data_non_stim.x.numpy()  # Retrieve the PSD
            psd_non_stim_file.append(psd_non_stim)

        # Convert lists to numpy arrays for easier manipulation
        psd_stim_file = np.array(psd_stim_file)  # Shape: [num_epochs, num_channels, num_frequencies]
        psd_non_stim_file = np.array(psd_non_stim_file)  # Shape: [num_epochs, num_channels, num_frequencies]

        # Calculate average and standard deviation for stimulation ON and OFF for the current file
        avg_psd_stim_file = np.mean(psd_stim_file, axis=0)  # Average across epochs
        std_psd_stim_file = np.std(psd_stim_file, axis=0)    # Standard deviation across epochs
        avg_psd_non_stim_file = np.mean(psd_non_stim_file, axis=0)  # Average across epochs
        std_psd_non_stim_file = np.std(psd_non_stim_file, axis=0)    # Standard deviation across epochs

        # Average over all channels for the current file
        avg_psd_stim_file_overall = np.mean(avg_psd_stim_file, axis=0)  # Average across channels
        std_psd_stim_file_overall = np.mean(std_psd_stim_file, axis=0)  # Average standard deviation across channels
        avg_psd_non_stim_file_overall = np.mean(avg_psd_non_stim_file, axis=0)  # Average across channels
        std_psd_non_stim_file_overall = np.mean(std_psd_non_stim_file, axis=0)  # Average standard deviation across channels

        # Define the frequency axis for the current file
        freqs = np.linspace(1, fmax, avg_psd_stim_file.shape[1])  # Shape: [num_frequencies]

        print('std PSDstim', std_psd_stim_file_overall)

        # Plot the average PSD for the current file
        plt.figure(figsize=(12, 5))
        
        # Plot for stimulation ON
        plt.plot(freqs, avg_psd_stim_file_overall, label='Average PSD (Stimulation ON)', color='blue')
        plt.fill_between(freqs, avg_psd_stim_file_overall - std_psd_stim_file_overall, avg_psd_stim_file_overall + std_psd_stim_file_overall, color='blue', alpha=0.2)

        # Plot for stimulation OFF
        plt.plot(freqs, avg_psd_non_stim_file_overall, label='Average PSD (Stimulation OFF)', color='red')
        plt.fill_between(freqs, avg_psd_non_stim_file_overall - std_psd_non_stim_file_overall, avg_psd_non_stim_file_overall + std_psd_non_stim_file_overall, color='red', alpha=0.2)

        # Set plot limits and labels
        plt.xlim(0, fmax)
        # plt.yscale("log")
        plt.title(f'Average Power Spectral Density for File {file_index + 1}')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.tight_layout() 
        plt.show()

def plot_PSD_avg(dataset, fmax):
    '''
    Plots the average power spectral density (PSD) of graphs where stimulation was turned ON and where stimulation was turned OFF.
    
    INPUT:
        - dataset       : Dataset of graphs
        - fmax          : Maximum frequency to plot
    OUTPUT: N/A
    '''

    # Retrieve how many graphs there are in the dataset
    total_per_file = dataset.graphs_per_file()  # Get tot total number of graphs per file
    print('total per file', total_per_file)
    stim_on_per_file = dataset.stim_graphs_per_file()  # Get the number of stim ON epochs per file
    stim_off_per_file = dataset.non_stim_graphs_per_file()  # Get the number of stim OFF epochs per file
    print('stim on per file', stim_on_per_file)
    print('stim off per file', stim_off_per_file)

    # Define the resampling frequency that is used in the 'process' function of 'MEGGraphs'
    sfreq = 256

    # Initialize lists to store average PSDs for each file
    avg_psd_stim_files = []
    avg_psd_non_stim_files = []

    # Loop through each file
    for file_index in range(len(total_per_file)):
        # Initialize lists to store PSD data for the current file
        all_psd_stim = []
        all_psd_non_stim = []

        # Loop through stimulation ON epochs
        for stim_index in range(stim_on_per_file[file_index]):
            print('graph', stim_index, 'file', file_index)
            data_stim = dataset.get(file_index, stim_index)
            if data_stim is not None:
                psd_stim = data_stim.x.numpy()  # Retrieve the PSD for each epoch in one file
                all_psd_stim.append(psd_stim)  # Append the PSD for each epoch to the list

        # Loop through stimulation OFF epochs
        for non_stim_index in range(stim_off_per_file[file_index]):
            data_non_stim = dataset.get(file_index, total_per_file[file_index] - 1 - non_stim_index)
            print('graph', non_stim_index, 'file', file_index)
            if data_non_stim is not None:
                print('data non stim', data_non_stim.x.shape)
                psd_non_stim = data_non_stim.x.numpy()  # Retrieve the PSD for each epoch in one file
                all_psd_non_stim.append(psd_non_stim)  # Append the PSD for each epoch to the list

        # Convert lists to numpy arrays for easier manipulation
        all_psd_stim = np.array(all_psd_stim)  # Shape: [num_epochs, num_channels, num_frequencies]
        all_psd_non_stim = np.array(all_psd_non_stim)  # Shape: [num_epochs, num_channels, num_frequencies]

        # Calculate average PSD across epochs for the current file
        avg_psd_stim_file = np.mean(all_psd_stim, axis=0)  # Average across epochs
        avg_psd_non_stim_file = np.mean(all_psd_non_stim, axis=0)  # Average across epochs

        # Store the average PSDs for this file
        avg_psd_stim_files.append(avg_psd_stim_file)
        avg_psd_non_stim_files.append(avg_psd_non_stim_file)

    print('Number of files:', len(avg_psd_stim_files))
    print('Shape of average PSD for stimulation ON:', avg_psd_stim_files[0].shape)

    # Convert lists to numpy arrays for overall averaging
    avg_psd_stim_files = np.array(avg_psd_stim_files)  # Shape: [num_files, num_channels, num_frequencies]
    avg_psd_non_stim_files = np.array(avg_psd_non_stim_files)  # Shape: [num_files, num_channels, num_frequencies]

    # Calculate overall average and standard deviation for stimulation ON and OFF
    overall_avg_psd_stim = np.mean(avg_psd_stim_files, axis=0)  # Average across files
    overall_std_psd_stim = np.std(avg_psd_stim_files, axis=0)  # Standard deviation across files
    overall_avg_psd_non_stim = np.mean(avg_psd_non_stim_files, axis=0)  # Average across files
    overall_std_psd_non_stim = np.std(avg_psd_non_stim_files, axis=0)  # Standard deviation across files

    # Average over all channels
    avg_psd_stim_overall = np.mean(overall_avg_psd_stim, axis=0)  # Average across channels
    std_psd_stim_overall = np.mean(overall_std_psd_stim, axis=0)  # Average standard deviation across channels
    avg_psd_non_stim_overall = np.mean(overall_avg_psd_non_stim, axis=0)  # Average across channels
    std_psd_non_stim_overall = np.mean(overall_std_psd_non_stim, axis=0)  # Average standard deviation across channels

    # Define the frequency axis, limited to fmax
    freqs = np.linspace(0, fmax, overall_avg_psd_stim.shape[1])  # Shape: [num_frequencies]
    print('Number of frequencies:', len(freqs))

    print('std PSDstim', std_psd_stim_overall)

    # Plot the average PSD for the current file
    plt.figure(figsize=(12, 5))

    # Plot for stimulation ON
    plt.plot(freqs, avg_psd_stim_overall, label='Average PSD (Stimulation ON)', color='blue')
    plt.fill_between(freqs, avg_psd_stim_overall - std_psd_stim_overall, avg_psd_stim_overall + std_psd_stim_overall, color='blue', alpha=0.2)

    # Plot for stimulation OFF
    plt.plot(freqs, avg_psd_non_stim_overall, label='Average PSD (Stimulation OFF)', color='red')
    plt.fill_between(freqs, avg_psd_non_stim_overall - std_psd_non_stim_overall, avg_psd_non_stim_overall + std_psd_non_stim_overall, color='red', alpha=0.2)

    # Set plot limits and labels
    plt.xlim(0, fmax)
    # plt.yscale("log")
    plt.title(f'Average Power Spectral Density for File {file_index + 1}')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.tight_layout() 
    plt.show()

    # Create a figure with subplots for each node
    num_nodes = overall_avg_psd_stim.shape[0]
    print('Number of nodes:', num_nodes)
    num_cols = 6  # Number of columns for subplots
    num_rows = (num_nodes + num_cols - 1) // num_cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 15), sharex=True, sharey=True)
    axes = axes.flatten()

    # Load the raw data to retrieve channel names
    raw = dataset.load_raw_data(dataset.raw_paths[0])

    # Retrieve the channel names
    channels = raw.info["ch_names"]

    # Plot the average PSD for each node
    for node in range(num_nodes):
        label = channels[node]
        ax = axes[node]
        
        # Plot for stimulation ON
        ax.plot(freqs, overall_avg_psd_stim[node], label='Stimulation ON', color='blue')
        ax.fill_between(freqs, overall_avg_psd_stim[node] - overall_std_psd_stim[node], overall_avg_psd_stim[node] + overall_std_psd_stim[node], color='blue', alpha=0.2)

        # Plot for stimulation OFF
        ax.plot(freqs, overall_avg_psd_non_stim[node], label='Stimulation OFF', color='red')
        ax.fill_between(freqs, overall_avg_psd_non_stim[node] - overall_std_psd_non_stim[node], overall_avg_psd_non_stim[node] + overall_std_psd_non_stim[node], color='red', alpha=0.2)

        # Set plot limits and labels
        ax.set_xlim(0, fmax)
        # ax.set_yscale("log")
        # ax.set_ylim(-0.5, 2)  # Adjust the y-axis limit to zoom in
        ax.set_title(f'{label}')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Magnitude')

    # Add a legend to the first subplot
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_PSD(dataset, fmax, file_index=0):
    '''
    Plots the power spectral density (PSD) of a graph where stimulation was turned ON and of a graph where stimulation was turned OFF.
    
    INPUT:
        - dataset       : Dataset of graphs
    OUTPUT: N/A
    '''

    # Retrieve how many graphs there are in the dataset
    total_per_file = dataset.graphs_per_file()

    # Retrieve a graph where stimulation was turned ON and a graph where stimulation was turned OFF. The first half of the created graphs are ON, the second half OFF. 
    data_stim = dataset.get(file_index, 0)
    data_non_stim = dataset.get(file_index, total_per_file[file_index]-1)

    # Define the resampling frequency that is used in the 'process' function of 'MEGGraphs' (see dataset.py)
    sfreq = 256

    # Retrieve the node feature matrices from both graphs (this is the PSD)
    psd_stim = data_stim.x.numpy()
    print('psd_stim', psd_stim.shape)
    psd_non_stim = data_non_stim.x.numpy()

    # Define the frequency axis, limited to fmax
    freqs = np.linspace(0, fmax, psd_stim.shape[1])

    # Limit the frequency axis to fmax
    freqs = freqs[freqs <= fmax]
    print(len(freqs))

    # Load the raw data 
    raw = dataset.load_raw_data(dataset.raw_paths[0])

    # Retrieve the channel names
    channels = raw.info["ch_names"]
    channels = [ch[:3] for ch in channels]

    # Plot the PSD of the stimulation ON graph
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    for idx in range(psd_stim.shape[0]):
        plt.plot(freqs, psd_stim[idx,:], label=channels[idx])

    plt.figlegend(channels)
    # plt.xlim(0, fmax)
    plt.yscale("log")
    plt.title('Power spectral density (stimulation ON)')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')

    # Plot the PSD of the stimulation OFF graph
    plt.subplot(122)
    for idx in range(psd_non_stim.shape[0]):
        plt.plot(freqs, psd_non_stim[idx,:], label=channels[idx])

    plt.figlegend(channels[:2])
    # plt.xlim(0, fmax)
    plt.yscale("log")
    plt.title('Power spectral density (stimulation OFF)')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')

    plt.tight_layout()
    plt.show()

def visualize_graph(dataset):
    '''
    Visualizes the network of one of the graphs in the dataset.

    INPUT:
        - dataset       : Dataset of graphs
    OUTPUT: N/A
    '''

    # Retrieve the first graph
    data = dataset.get(0,0)
    print('data graph', data)

    # Load the raw data
    raw = dataset.load_raw_data(dataset.raw_paths[0])
    print('raw', raw.info)

    # Retrieve the channel names
    channels = raw.info["ch_names"]
    channels_dict = {n:channel[:3] for n, channel in enumerate(channels)}

    # Retrieve the channel positions
    ch_pos = [ch['loc'][:2] for ch in raw.info['chs']]
    ch_pos_dict = {idx: pos for idx, pos in enumerate(ch_pos)}

    # Convert the graph to a NetworkX graph
    g = torch_geometric.utils.to_networkx(data, to_undirected=True)

    # Retrieve edge weights and normalize for visualization
    edge_weights = data.edge_attr.numpy()
    edge_weights = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min())

    # Visualize graph with edge weights
    plt.figure()
    nx.draw_networkx(g, pos=ch_pos_dict, with_labels=False, node_color='r', edge_color='k', node_size=450, width=edge_weights * 5)
    nx.draw_networkx_labels(g, pos=ch_pos_dict, labels=channels_dict, font_color='k', font_weight='bold', font_size=9)
    plt.show()

def visualize_filtered_graph(filtered_graph, input_type='fif', scouts_data_list=None):
    '''
    Visualizes the network of one of the graphs in the dataset after edge filtering.

    INPUT:
        - filtered_graph    : One graph object after filtering with best performing hyperparameters
        - input_type        : Type of input data ('fif' or 'scout')
        - scouts_data_list  : List of scout data objects (required if input_type is 'scout')
    OUTPUT: N/A
    '''
    	
    if input_type == 'fif':
        # Retrieve the raw filename of the first graph
        directory = r'F:\MEG GNN\GNN\Data\Raw'
        filenames = []
        for filename in os.listdir(directory):
            if filename.endswith('.fif'):
                filenames.append(filename)
        
        filename = filenames[0]
        file_path = os.path.join(r'F:\MEG GNN\GNN\Data\Raw', filename)
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

        # Retrieve the channel names
        channels = raw.info["ch_names"]
        channels_dict = {n:channel[:3] for n, channel in enumerate(channels)}

        # Retrieve the channel positions
        ch_pos = [ch['loc'][:2] for ch in raw.info['chs']]
        ch_pos_dict = {idx: pos for idx, pos in enumerate(ch_pos)}

    elif input_type == 'scout':
        if scouts_data_list is None:
            raise ValueError("scouts_data_list must be provided when input_type is 'scout'")

        # Retrieve the scout data for the first file
        scouts_data = scouts_data_list[0]
        scout_names = list(scouts_data.keys())
        scout_positions = {idx: np.random.rand(2) for idx in range(len(scout_names))}  # Random positions for visualization

        # Create a dictionary for scout names
        channels_dict = {idx: scout for idx, scout in enumerate(scout_names)}
        ch_pos_dict = scout_positions

    else:
        raise ValueError("Invalid input_type. Must be 'fif' or 'scout'.")

    # Convert the graph to a NetworkX graph
    g = torch_geometric.utils.to_networkx(filtered_graph, to_undirected=True)

    # Retrieve edge weights and normalize for visualization
    edge_weights = filtered_graph.edge_attr.numpy()
    edge_weights = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min())

    # Visualize graph with edge weights
    plt.figure()
    nx.draw_networkx(g, pos=ch_pos_dict, with_labels=False, node_color='r', edge_color='k', node_size=450, width=edge_weights * 5)
    nx.draw_networkx_labels(g, pos=ch_pos_dict, labels=channels_dict, font_color='k', font_weight='bold', font_size=9)
    plt.show()

def plot_connectivity_matrix(dataset, file_index=0):
    '''
    Plots the connectivity matrix (heatmap) for a given file.

    INPUT:
        - dataset       : Dataset of graphs
        - file_index    : Index of the file to plot the connectivity matrix for (default is 0)
    
    OUTPUT: N/A
    '''

    # Retrieve the graph for the specified file index
    data = dataset.get(file_index, 0)

    # Retrieve the edge index and edge weight (connectivity values)
    edge_index = data.edge_index.numpy()
    edge_weight = data.edge_attr.numpy()

    # Load the raw data to retrieve channel names
    raw = dataset.load_raw_data(dataset.raw_paths[file_index])
    channels = raw.info["ch_names"]

    # Create an empty connectivity matrix
    num_channels = len(channels)
    connectivity_matrix = np.zeros((num_channels, num_channels))

    # Fill the connectivity matrix with edge weights
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        connectivity_matrix[src, dst] = edge_weight[i]
        connectivity_matrix[dst, src] = edge_weight[i]

    # Plot the connectivity matrix as a heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(connectivity_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Connectivity Strength')
    plt.xticks(ticks=np.arange(num_channels), labels=channels, rotation=90)
    plt.yticks(ticks=np.arange(num_channels), labels=channels)
    plt.title(f'Connectivity Matrix for File {file_index + 1}')
    plt.tight_layout()
    plt.show()

def visualize_scout_graph(dataset, scouts_data_list):
    '''
    Visualizes the network of one of the graphs in the dataset for scout data.

    INPUT:
        - dataset           : Dataset of graphs
        - scouts_data_list  : List of dictionaries holding scouts data for all files
    OUTPUT: N/A
    '''

    # Retrieve the first graph
    data = dataset.get(0, 0)
    print('data graph', data)

    # Retrieve the scout data for the first file
    scouts_data = scouts_data_list[0]
    scout_names = list(scouts_data.keys())
    print('scout names', scout_names)

    # Load the scout coordinates from the Excel file
    scout_coords_file = r'F:\MEG GNN\GNN\Data\Raw\Scout files\Mat files\Scout MNI coordinates.xlsx'
    scout_coords_df = pd.read_excel(scout_coords_file)

    # Create a dictionary for scout positions
    scout_positions = {}
    for idx, scout in enumerate(scout_names):
        scout_row = scout_coords_df[scout_coords_df['Scout'] == scout]
        if not scout_row.empty:
            x, y, z = scout_row.iloc[0, 1:4]
            scout_positions[idx] = (x, y)
        else:
            scout_positions[idx] = np.random.rand(2)  # Random position if scout not found
    
    print('scout positions', scout_positions)
    
    # Create a dictionary for scout names
    scouts_dict = {idx: scout for idx, scout in enumerate(scout_names)}

    # Convert the graph to a NetworkX graph
    g = torch_geometric.utils.to_networkx(data, to_undirected=True)

    # Retrieve edge weights and normalize for visualization
    edge_weights = data.edge_attr.numpy()
    edge_weights = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min())

    # Visualize graph with edge weights
    plt.figure()
    nx.draw_networkx(g, pos=scout_positions, with_labels=False, node_color='r', edge_color='k', node_size=450, width=edge_weights * 5)
    nx.draw_networkx_labels(g, pos=scout_positions, labels=scouts_dict, font_color='k', font_weight='bold', font_size=9)
    plt.show()

def plot_scout_connectivity_matrix(dataset, scouts_data_list, file_index=0):
    '''
    Plots the connectivity matrix (heatmap) for a given file for scout data.

    INPUT:
        - dataset           : Dataset of graphs
        - scouts_data_list  : List of dictionaries holding scouts data for all files
        - file_index        : Index of the file to plot the connectivity matrix for (default is 0)
    
    OUTPUT: N/A
    '''

    # Retrieve the graph for the specified file index
    data = dataset.get(file_index, 0)

    # Retrieve the edge index and edge weight (connectivity values)
    edge_index = data.edge_index.numpy()
    edge_weight = data.edge_attr.numpy()

    # Retrieve the scout data for the specified file index
    scouts_data = scouts_data_list[file_index]
    scout_names = list(scouts_data.keys())

    # Create an empty connectivity matrix
    num_scouts = len(scout_names)
    connectivity_matrix = np.zeros((num_scouts, num_scouts))

    # Fill the connectivity matrix with edge weights
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        connectivity_matrix[src, dst] = edge_weight[i]
        connectivity_matrix[dst, src] = edge_weight[i]

    # Plot the connectivity matrix as a heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(connectivity_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Connectivity Strength')
    plt.xticks(ticks=np.arange(num_scouts), labels=scout_names, rotation=90)
    plt.yticks(ticks=np.arange(num_scouts), labels=scout_names)
    plt.title(f'Connectivity Matrix for File {file_index + 1}')
    plt.tight_layout()
    plt.show()

def plot_PSD_avg_scouts(dataset, scouts_data_list, fmax):
    '''
    Plots the average power spectral density (PSD) of graphs where stimulation was turned ON and where stimulation was turned OFF for scout data.
    
    INPUT:
        - dataset           : Dataset of graphs
        - scouts_data_list  : List of dictionaries holding scouts data for all files
        - fmax              : Maximum frequency to plot
    OUTPUT: N/A
    '''

    # Retrieve how many graphs there are in the dataset
    total_per_file = dataset.graphs_per_file()  # Get total number of graphs per file
    stim_on_per_file = dataset.stim_graphs_per_file()  # Get the number of stim ON epochs per file
    stim_off_per_file = dataset.non_stim_graphs_per_file()  # Get the number of stim OFF epochs per file

    # Initialize lists to store average PSDs for each file
    avg_psd_stim_files = []
    avg_psd_non_stim_files = []

    # Loop through each file
    for file_index in range(len(total_per_file)):
        # Initialize lists to store PSD data for the current file
        all_psd_stim = []
        all_psd_non_stim = []

        # Loop through stimulation ON epochs
        for stim_index in range(stim_on_per_file[file_index]):
            data_stim = dataset.get(file_index, stim_index)
            psd_stim = data_stim.x.numpy()  # Retrieve the PSD for each epoch in one file
            all_psd_stim.append(psd_stim)  # Append the PSD for each epoch to the list

        # Loop through stimulation OFF epochs
        for non_stim_index in range(stim_off_per_file[file_index]):
            data_non_stim = dataset.get(file_index, total_per_file[file_index] - 1 - non_stim_index)
            psd_non_stim = data_non_stim.x.numpy()  # Retrieve the PSD for each epoch in one file
            all_psd_non_stim.append(psd_non_stim)  # Append the PSD for each epoch to the list

        # Convert lists to numpy arrays for easier manipulation
        all_psd_stim = np.array(all_psd_stim)  # Shape: [num_epochs, num_channels, num_frequencies]
        all_psd_non_stim = np.array(all_psd_non_stim)  # Shape: [num_epochs, num_channels, num_frequencies]

        # Calculate average PSD across epochs for the current file
        avg_psd_stim_file = np.mean(all_psd_stim, axis=0)  # Average across epochs
        avg_psd_non_stim_file = np.mean(all_psd_non_stim, axis=0)  # Average across epochs

        # Store the average PSDs for this file
        avg_psd_stim_files.append(avg_psd_stim_file)
        avg_psd_non_stim_files.append(avg_psd_non_stim_file)

    print('Number of files:', len(avg_psd_stim_files))
    print('Shape of average PSD for stimulation ON:', avg_psd_stim_files[0].shape)

    # Convert lists to numpy arrays for overall averaging
    avg_psd_stim_files = np.array(avg_psd_stim_files)  # Shape: [num_files, num_channels, num_frequencies]
    avg_psd_non_stim_files = np.array(avg_psd_non_stim_files)  # Shape: [num_files, num_channels, num_frequencies]

    # Calculate overall average and standard deviation for stimulation ON and OFF
    overall_avg_psd_stim = np.mean(avg_psd_stim_files, axis=0)  # Average across files
    overall_std_psd_stim = np.std(avg_psd_stim_files, axis=0)  # Standard deviation across files
    overall_avg_psd_non_stim = np.mean(avg_psd_non_stim_files, axis=0)  # Average across files
    overall_std_psd_non_stim = np.std(avg_psd_non_stim_files, axis=0)  # Standard deviation across files

    # Average over all channels
    avg_psd_stim_overall = np.mean(overall_avg_psd_stim, axis=0)  # Average across channels
    std_psd_stim_overall = np.mean(overall_std_psd_stim, axis=0)  # Average standard deviation across channels
    avg_psd_non_stim_overall = np.mean(overall_avg_psd_non_stim, axis=0)  # Average across channels
    std_psd_non_stim_overall = np.mean(overall_std_psd_non_stim, axis=0)  # Average standard deviation across channels

    # Define the frequency axis, limited to fmax
    freqs = np.linspace(0, fmax, overall_avg_psd_stim.shape[1])  # Shape: [num_frequencies]
    print('Number of frequencies:', len(freqs))

    print('std PSDstim', std_psd_stim_overall)

    # Plot the average PSD for the current file
    plt.figure(figsize=(12, 5))

    # Plot for stimulation ON
    plt.plot(freqs, avg_psd_stim_overall, label='Average PSD (Stimulation ON)', color='blue')
    plt.fill_between(freqs, avg_psd_stim_overall - std_psd_stim_overall, avg_psd_stim_overall + std_psd_stim_overall, color='blue', alpha=0.2)

    # Plot for stimulation OFF
    plt.plot(freqs, avg_psd_non_stim_overall, label='Average PSD (Stimulation OFF)', color='red')
    plt.fill_between(freqs, avg_psd_non_stim_overall - std_psd_non_stim_overall, avg_psd_non_stim_overall + std_psd_non_stim_overall, color='red', alpha=0.2)

    # Set plot limits and labels
    plt.xlim(0, fmax)
    # plt.yscale("log")
    plt.title(f'Average Power Spectral Density for All Files')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.tight_layout() 
    plt.show()

    # Create a figure with subplots for each scout
    num_scouts = overall_avg_psd_stim.shape[0]
    print('Number of scouts:', num_scouts)
    num_cols = 6  # Number of columns for subplots
    num_rows = (num_scouts + num_cols - 1) // num_cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 15), sharex=True, sharey=True)
    axes = axes.flatten()

    # Retrieve the scout names
    scouts_data = scouts_data_list[0]
    scout_names = list(scouts_data.keys())

    # Plot the average PSD for each scout
    for scout in range(num_scouts):
        label = scout_names[scout]
        ax = axes[scout]
        
        # Plot for stimulation ON
        ax.plot(freqs, overall_avg_psd_stim[scout], label='Stimulation ON', color='blue')
        ax.fill_between(freqs, overall_avg_psd_stim[scout] - overall_std_psd_stim[scout], overall_avg_psd_stim[scout] + overall_std_psd_stim[scout], color='blue', alpha=0.2)

        # Plot for stimulation OFF
        ax.plot(freqs, overall_avg_psd_non_stim[scout], label='Stimulation OFF', color='red')
        ax.fill_between(freqs, overall_avg_psd_non_stim[scout] - overall_std_psd_non_stim[scout], overall_avg_psd_non_stim[scout] + overall_std_psd_non_stim[scout], color='red', alpha=0.2)

        # Set plot limits and labels
        ax.set_xlim(0, fmax)
        # ax.set_yscale("log")
        # ax.set_ylim(-0.5, 2)  # Adjust the y-axis limit to zoom in
        ax.set_title(f'{label}')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Magnitude')

    # Add a legend to the first subplot
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    # Adjust layout
    plt.tight_layout()
    plt.show()
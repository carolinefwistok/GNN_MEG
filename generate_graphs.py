import os
import re
import shutil
import argparse
from mpi4py import MPI
from data_utils import *
from dataset import MEGGraphs

def filter_filenames(filenames, analysis):
    '''
    Filters filenames based on the analysis type.
    INPUTS:
        - filenames:    List of filenames to filter.
        - analysis:     Analysis type to filter by (e.g., 'BURST', 'TONIC', 'Canada', 'Nijmegen').
    OUTPUT:
        - filenames:    List of filtered filenames.
    '''

    # Convert analysis to uppercase for case-insensitive comparison
    analysis = analysis.upper()
    if analysis == "BURST":
        return [f for f in filenames if "BURST" in f.upper()]
    elif analysis == "TONIC":
        return [f for f in filenames if "TONIC" in f.upper()]
    elif analysis == "CANADA":
        return [f for f in filenames if re.search(r'\bPT\d', f.upper())]
    elif analysis == "NIJMEGEN":
        return [f for f in filenames if re.match(r'^PTN\d', f.upper())]
    else:
        return filenames

def get_filenames(input_type='scout', scout_dir=None):
    '''
    Retrieves filenames based on the input type.
    INPUT:
        - input_type:   Type of input data, either 'fif' or 'scout'.
    OUTPUT:
        - filenames:    List of filenames to process.
    '''

    # Define the root directory and get filenames based on input type
    root_directory = '/scratch/cwitstok/Data'
    if input_type == 'fif':
        raw_dir = os.path.join(root_directory, "Raw_all")
        filenames = sorted([f for f in os.listdir(raw_dir) if f.endswith('.fif')])
    elif input_type == 'scout':
        # scout_dir = os.path.join(root_directory, "Scout_all")
        scout_files = sorted([f for f in os.listdir(scout_dir) if f.endswith('.h5')])
        fif_dir = os.path.join(root_directory, "Raw_all")
        filenames = get_raw_file(scout_files, fif_dir)
    else:
        raise ValueError(f'Input type {input_type} not recognized')
    return filenames

def process_single_file(fname, input_type, root_directory, processed_analysis_dir, scout_dir, fmin=1, fmax=100):
    '''
    Processes a single file to generate MEG graphs.
    INPUTS:
        - fname:                    Name of the file to process.
        - input_type:               Type of input data, either 'fif' or 'scout
        - root_directory:           Root directory where the data is stored.
        - processed_analysis_dir:   Directory where the processed data will be stored.
        - scout_dir:                Directory where the scout data is stored.
        - fmin:                     Minimum frequency for connectivity analysis.
        - fmax:                     Maximum frequency for connectivity analysis.
    OUTPUT: N/A
    '''

    print(f"Processing file: {fname} with input type: {input_type}")
    # Prepare stimulation info
    stim_excel_file = '/scratch/cwitstok/Data/MEG_PT_notes.xlsx'
    stim_info = get_stimulation_info(stim_excel_file, fname)
    stim_info_dict = {fname: stim_info}

    # Prepare scout data if needed
    scouts_data_list = None
    if input_type == 'scout':
        prefix = fname.split('_FT_data_raw')[0]
        scout_candidates = [f for f in os.listdir(scout_dir) if f.startswith(prefix) and f.endswith('.h5')]
        if not scout_candidates:
            print(f"No scout file found for {fname}, skipping.")
            return
        scouts_data_list = [load_scouts_data([scout_candidates[0]], scout_dir)[0]]

    # Set threshold based on scout_dir
    if "Scout_33" in scout_dir:
        threshold_scout = 100
        scout_version = "Scout_v33"
    elif "Scout_58" in scout_dir and not "Scout_58_MN" in scout_dir:
        threshold_scout = 40
        scout_version = "Scout_v58"
    elif "Scout_58_MN" in scout_dir:
        threshold_scout = 120
        scout_version = "Scout_v58_MN"
    else:
        threshold_scout = 100

    # Set parameters for MEGGraphs
    duration = 30
    overlap = 25
    ramp_time = 5
    conn_method = 'pli'
    freq_res = 1
    conn_save_dir = f'/scratch/cwitstok/Data/Connectivity/{scout_version}_{duration}sec_{overlap}overlap_freq_{fmin}_{fmax}Hz_freqres_{freq_res}Hz'

    # Process the file
    dataset = MEGGraphs(
        input_type=input_type,
        root=root_directory,
        filenames=[fname],
        stim_info_dict=stim_info_dict,
        duration=duration,
        overlap=overlap,
        conn_method=conn_method,
        fmin=fmin,
        fmax=fmax,
        ramp_time=ramp_time,
        conn_save_dir=conn_save_dir,
        freq_res=freq_res,
        scout_data_list=scouts_data_list,
        processed_dir=processed_analysis_dir,
        threshold_scout=threshold_scout
    )

    # dataset.remove_bad_graphs() 

if __name__ == "__main__":
    ''' Main entry point for the script. Initializes MPI, sets up directories, and processes files based on analysis type. '''
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate MEG graphs for analysis.")
    parser.add_argument("--processed_dir", type=str, required=True, help="Path to processed directory.")
    parser.add_argument("--scout_version", type=str, default="Scout_all", help="Scout folder to use (e.g., Scout_33, Scout_58, Scout_all)")
    args = parser.parse_args()

    # Set input type and directories
    input_type = 'scout'
    root_directory = '/scratch/cwitstok/Data'
    # processed_dir = os.path.join(root_directory, "processed")

    # Get root directory and scout version from environment variables or defaults
    processed_dir = args.processed_dir
    scout_dir = os.path.join(root_directory, "raw", "Scout_files", args.scout_version)

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Get analysis name from environment variable or default to Full
    analysis = os.environ.get("ANALYSIS_NAME", "Full")
    print(f"Running analysis: {analysis}")

    # Define frequency bands
    frequency_bands = {
        "delta":  (1, 4),
        "theta":  (4, 8),
        "alpha":  (8, 12),
        "beta":   (12, 30),
        "gamma":  (30, 100),
        "low": (1, 30),
    }

    # Decide which files and frequency range to use
    all_filenames = get_filenames(input_type=input_type, scout_dir=scout_dir)
    if analysis.lower() in frequency_bands:
        filtered_filenames = all_filenames
        fmin, fmax = frequency_bands[analysis.lower()]
    else:
        filtered_filenames = filter_filenames(all_filenames, analysis)
        fmin, fmax = 1, 100  # Default for non-frequency analyses
        for f in filtered_filenames:
            print(f"Filtered file for analysis {analysis}: {f}")

    # Distribute filtered files among ranks
    num_files = len(filtered_filenames)
    files_for_this_rank = [filtered_filenames[i] for i in range(num_files) if i % size == rank]

    # Create processed directory for this analysis
    processed_dir_analysis = os.path.join(processed_dir, f'processed_{analysis}')
    print(f"Rank {rank} will process files in: {processed_dir_analysis}")

    # Save bad subepochs to a subfolder inside the analysis folder
    bad_subepochs_dir = os.path.join(processed_dir_analysis, "bad_subepochs")
    if rank == 0:
        # Only delete if the processed_dir_analysis itself is new/empty
        if not os.path.exists(processed_dir_analysis):
            os.makedirs(processed_dir_analysis, exist_ok=True)
        os.makedirs(bad_subepochs_dir, exist_ok=True)
        print(f"Bad subepochs will be saved in: {bad_subepochs_dir}")

    # Process each file separately
    for fname in files_for_this_rank:
        print(f"Rank {rank} processing file: {fname}")
        process_single_file(fname, input_type, root_directory, processed_dir_analysis, scout_dir=scout_dir, fmin=fmin, fmax=fmax)
    
    # Process all files in the current rank
    # for fname in filtered_filenames:
    #     print(f"Rank {rank} processing file: {fname}")
    #     process_single_file(fname, input_type, root_directory, processed_dir_analysis, scout_dir=scout_dir, fmin=fmin, fmax=fmax)

    comm.Barrier()

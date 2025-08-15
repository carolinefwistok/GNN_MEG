import os
import glob
import shutil

def clear_folder(folder_path):
    '''
    Clears the specified folder by removing all files and subdirectories.
    Inputs:
        - folder_path: Path to the folder to clear.
    OUTPUT: N/A
    '''

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
            print(f"Removed {file_path}")
        except Exception as e:
            print(f"Could not remove {file_path}: {e}")

def ensure_graphs_for_experiment(experiment):
    """
    Ensures that the processed graphs for the specified experiment exist in the Processed folder.
    If the graphs already exist, it simply copies them to the Processed_saved\<experiment_name> folder.

    Inputs:
        - experiment: Name of the experiment, e.g. 'tonic', 'burst', 'canada', 'nijmegen', 'full', or other frequency bands.
    OUTPUT: N/A
    """
    processed_saved_dir = os.path.join(r"F:\MEG_GNN\GNN\Data\Processed_saved", experiment, 'all_graphs')
    processed_dir = r"F:\MEG_GNN\GNN\Data\Processed"

    # Check if Processed_saved/<experiment> exists and contains graph files
    has_graphs = False
    if os.path.exists(processed_saved_dir):
        for root, dirs, files in os.walk(processed_saved_dir):
            if any(f.endswith('.pt') for f in files):
                has_graphs = True
                break

    if has_graphs:
        print(f"Graphs already exist for {experiment} in {processed_saved_dir}. Copying to {processed_dir}...")
        # Remove current Processed directory and copy saved graphs
        if os.path.exists(processed_dir):
            shutil.rmtree(processed_dir)
        shutil.copytree(processed_saved_dir, processed_dir)
        print("Copied graphs to Processed.")
    else:
        print(f"No graphs found for {experiment}. Please create them first!")

def save_graphs_for_experiment(experiment):
    """
    Copies the current Processed folder to Processed_saved/<experiment>.

    Inputs:
        - experiment: Name of the experiment, e.g. 'tonic', 'burst', 'canada', 'nijmegen', 'full', or other frequency bands.
    OUTPUT: N/A
    """
    processed_dir = r"F:\MEG_GNN\GNN\Data\Processed"
    processed_saved_dir = os.path.join(r"F:\MEG_GNN\GNN\Data\Processed_saved", experiment)

    # Remove existing saved folder if it exists
    if os.path.exists(processed_saved_dir):
        shutil.rmtree(processed_saved_dir)
    shutil.copytree(processed_dir, processed_saved_dir)
    print(f"Copied graphs to {processed_saved_dir}")

def fix_processed_folder(experiment):
    '''
    Sets up the working Processed folder for the specified experiment.
    Copies all contents (including all_graphs and bad_subepochs) from the source processed folder.
    Inputs:
        - experiment: Name of the experiment.
    OUTPUT: N/A
    '''

    # Map experiment to the correct processed folder
    processed_map = {
        'tonic': r"F:\MEG_GNN\GNN\Data\Processed_saved\TONIC",
        'burst': r"F:\MEG_GNN\GNN\Data\Processed_saved\BURST",
        'canada': r"F:\MEG_GNN\GNN\Data\Processed_saved\Canada",
        'nijmegen': r"F:\MEG_GNN\GNN\Data\Processed_saved\Nijmegen",
        'full': r"F:\MEG_GNN\GNN\Data\Processed_saved\Full",
        'delta': r"F:\MEG_GNN\GNN\Data\Processed_saved\Delta",
        'theta': r"F:\MEG_GNN\GNN\Data\Processed_saved\Theta",
        'alpha': r"F:\MEG_GNN\GNN\Data\Processed_saved\Alpha",
        'theta_alpha': r"F:\MEG_GNN\GNN\Data\Processed_saved\Theta_Alpha",
        'beta': r"F:\MEG_GNN\GNN\Data\Processed_saved\Beta",
        'gamma': r"F:\MEG_GNN\GNN\Data\Processed_saved\Gamma",
    }
    src_folder = processed_map.get(experiment.lower())
    if src_folder is None:
        raise ValueError(f"Unknown experiment type: {experiment}")

    # Set the destination folder
    dst_folder = r"F:\MEG_GNN\GNN\Data\Processed"

    # Clear the destination folder
    if os.path.exists(dst_folder):
        shutil.rmtree(dst_folder)
        print(f"Removed '{dst_folder}'.")

    # Copy the entire processed folder (including subfolders)
    shutil.copytree(src_folder, dst_folder)
    print(f"Copied '{src_folder}' to '{dst_folder}'.")

def filter_files_for_experiment(files, experiment):
    '''
    Filters the list of files based on the specified experiment type.
    Inputs:
        - files:        List of filenames to filter.
        - experiment:   Experiment name, e.g. 'tonic', 'burst', 'canada', 'nijmegen', 'full', or other frequency bands.
    OUTPUT: List of filtered filenames
    '''

    # Filter files based on the experiment type
    exp = experiment.lower()
    if exp == 'tonic':
        return [f for f in files if 'TONIC' in f]
    elif exp == 'burst':
        return [f for f in files if 'BURST' in f]
    elif exp == 'canada':
        return [f for f in files if f.startswith('PT') and not f.startswith('PTN')]
    elif exp == 'nijmegen':
        return [f for f in files if f.startswith('PTN')]
    elif exp == 'full':
        return [f for f in files if f.startswith('PT') or f.startswith('PTN')]
    elif exp in ['delta', 'theta', 'alpha', 'theta_alpha', 'beta', 'gamma']:
        return [f for f in files if f.startswith('PT') or f.startswith('PTN')]
    else:
        raise ValueError(f"Unknown experiment type: {experiment}")

def fix_raw_files(experiment, input_type):
    '''
    Fixes the fif and scout maps for the specified experiment.
    Inputs:
        - experiment:        Experiment name, e.g. 'tonic', 'burst', 'canada', 'nijmegen', 'full', or other frequency bands.
        - input_type:        Input type of the data, either 'fif' or 'scout'.
    OUTPUT:
        - fif_folder:        Path to the folder containing the fif files.
        - scout_folder_used: Path to the folder containing the scout files (if input_type is 'scout').
    '''

    # Define the source and destination folders for fif and scout files
    fif_folder = r"F:\MEG_GNN\GNN\Data\Raw"
    raw_copy_folder = r"F:\MEG_GNN\GNN\Data\Raw_all"
    scout_folder = os.path.join(fif_folder, "Scout_files")
    scout_copy_folder = r"F:\MEG_GNN\GNN\Data\Scout_all"

    # Copy the fif files based on the experiment type
    all_raw_files = [f for f in os.listdir(raw_copy_folder) if os.path.isfile(os.path.join(raw_copy_folder, f))]
    files_to_copy = filter_files_for_experiment(all_raw_files, experiment)
    os.makedirs(fif_folder, exist_ok=True)
    clear_folder(fif_folder)
    for file in files_to_copy:
        src_path = os.path.join(raw_copy_folder, file)
        dst_path = os.path.join(fif_folder, file)
        try:
            shutil.copy2(src_path, dst_path)
            print(f"Copied {src_path} to {dst_path}")
        except Exception as e:
            print(f"Could not copy {file}: {e}")

    scout_folder_used = None
    # Copy the scout files based on the experiment type
    if input_type == 'scout':
        os.makedirs(scout_folder, exist_ok=True)
        clear_folder(scout_folder)
        all_scout_files = [f for f in os.listdir(scout_copy_folder) if os.path.isfile(os.path.join(scout_copy_folder, f))]
        scout_files_to_copy = filter_files_for_experiment(all_scout_files, experiment)
        for file in scout_files_to_copy:
            src_path = os.path.join(scout_copy_folder, file)
            dst_path = os.path.join(scout_folder, file)
            try:
                shutil.copy2(src_path, dst_path)
                print(f"Copied {src_path} to {dst_path}")
            except Exception as e:
                print(f"Could not copy {file}: {e}")
        scout_folder_used = scout_folder

    # Return the folders used so they can be passed to create_dataset
    return fif_folder, scout_folder_used

def get_graph_files(processed_dir, analysis):
    '''
    Returns a list of graph files in the processed directory based on the specified analysis type.
    Inputs:
        - processed_dir: Path to the processed directory containing graph files.
        - analysis:      Name of the analysis, e.g. 'tonic', 'burst', 'canada', 'nijmegen', 'full', or other frequency bands.
    OUTPUT: List of graph files matching the specified analysis type.
    '''

    # Determine the pattern for graph files based on the analysis type
    if analysis.lower() == 'burst':
        pattern = "graph_*_BURST_*.pt"
    elif analysis.lower() == 'tonic':
        pattern = "graph_*_TONIC_*.pt"
    elif analysis.lower() == 'canada':
        pattern = "graph_PT*_*.pt"
    elif analysis.lower() == 'nijmegen':
        pattern = "graph_PTN*_*.pt"
    elif analysis.lower() == 'full' or analysis.lower() == 'delta' or analysis.lower() == 'theta' or analysis.lower() == 'alpha' or analysis.lower() == 'theta_alpha' or analysis.lower() == 'beta' or analysis.lower() == 'gamma':
        pattern = "graph_*.pt"
    else:
        raise ValueError(f"Unknown analysis: {analysis}")
    
    # Use glob to find all files matching the pattern in the processed directory
    return glob.glob(os.path.join(processed_dir, pattern))
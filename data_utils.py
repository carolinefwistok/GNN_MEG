import pandas as pd
import h5py
import os

def get_stimulation_info(excel_file, filename):
    '''
    Retrieves stimulation information based on the filename.
    
    INPUT:
        - excel_file    : Path to the Excel file containing stimulation data.
        - filename      : The filename to extract the stimulation data from.
    
    OUTPUT:
        - stim_info     : Series containing the stimulation information for the specified file.
    '''

    # Read the Excel file
    df = pd.read_excel(excel_file, 'Information')

    # List of columns to convert to float
    float_columns = ['Stim_freq', 'First_OFF_time', 'Second_OFF_time', 'Last_OFF_time']
    
    # Convert specified columns to float
    df[float_columns] = df[float_columns].astype(float)

    # Account for excel file formatting (in Dutch Excel settings, a dot is not viewed as decimal point)
    df['First_OFF_time'] /= 1000
    df['Second_OFF_time'] /= 1000
    df['Last_OFF_time'] /= 1000

    # Extract the identifier from the filename
    identifier = filename.split('_')[0] + '_' + filename.split('_')[1].lower() + '_' + filename.split('_')[2]
    print(identifier)

    # Filter the DataFrame for the relevant row
    stim_info = df[df['File'].str.lower() == identifier.lower()]
    if not stim_info.empty:
        stim_info = stim_info.iloc[0].copy()  # Create a copy of the Series to modify
        for col in ['First_OFF_time', 'Second_OFF_time', 'Last_OFF_time']:
            stim_info[col] = f"{stim_info[col]:.3f}"  # Format to three decimal places
        return stim_info
    else:
        # If stim_info is not found, prompt the user to input the values manually
        print(f"No stimulation info found for '{filename}'. Please enter the details:")
        
        stim_freq = float(input("Enter Stim_freq: "))
        first_off_time = float(input("Enter First_OFF_time: "))
        second_off_time = float(input("Enter Second_OFF_time: "))
        last_off_time = float(input("Enter Last_OFF_time: "))

        # Create a Series with the manually entered data
        stim_info = pd.Series({
            'File': identifier,
            'Stim_freq': f"{stim_freq:.3f}",
            'First_OFF_time': f"{first_off_time:.3f}",
            'Second_OFF_time': f"{second_off_time:.3f}",
            'Last_OFF_time': f"{last_off_time:.3f}"
        })
        
        return stim_info

def load_scouts_data(scout_filenames, directory):
    '''
    Load scouts data from the specified files and save it to a list of dictionaries.

    INPUT:
        - scout_filenames   : list of paths to the scouts data files
        - directory         : path to the directory containing the scouts data files
    OUTPUT:
        - scouts_data_list  : list of dictionaries containing the scouts data for each file
    '''
    
    # Initialize list to store scouts data
    scouts_data_list = []

    # Load scouts data from the specified files
    for scout_filename in scout_filenames:
        print(f'Loading scouts data from {scout_filename}...')
        # Load scouts data from the specified file and save it in a dictionary
        scouts_data = {}
        with h5py.File(os.path.join(directory, scout_filename), 'r') as f:
            for scout_name in f.keys():
                # Extract name string from the scout name
                scout_name_str = scout_name.strip("[]").strip("'")
                print(f'Loading scout {scout_name_str}...')
                
                # Save the scout data to the dictionary
                scouts_data[scout_name_str] = f[scout_name][:]

        # Append the scouts data dictionary to the list
        scouts_data_list.append(scouts_data)
    
    return scouts_data_list

def get_raw_file(scout_filenames, fif_directory):
    '''
    Get the raw fif file corresponding to the scout data file.

    INPUT:
        - scout_filenames   : list of paths to the scouts data files
        - fif_directory     : path to the directory containing the raw fif files
    
    OUTPUT:
        - fif_files_list    : list of raw fif filenames corresponding to the scout data files
    '''

    # Initialize list to store fif filenames
    fif_files_list = []

    # Load the fif file corresponding to the scout file
    for scout_filename in scout_filenames:
        # Automatically generate the scout file prefix
        scout_file_prefix = '_'.join(scout_filename.split('_')[:3])

        # Find the corresponding fif file in the fif directory
        found = False
        for filename in os.listdir(fif_directory):
            if filename.startswith(scout_file_prefix) and filename.endswith('_raw.fif'):
                fif_files_list.append(filename)
                found = True
                break
        if not found:
            raise FileNotFoundError(f"No matching FIF file found for scout file prefix {scout_file_prefix}\n"
                                    f"Check the file directory {fif_directory}!")
    return fif_files_list
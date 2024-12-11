import pandas as pd

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

    # Account for excel file formatting (not seeing a dot as decimal points)
    df['First_OFF_time'] /= 1000
    df['Second_OFF_time'] /= 1000
    df['Last_OFF_time'] /= 1000

    # Extract the identifier from the filename
    identifier = filename.split('_')[0] + '_' + filename.split('_')[1].lower()

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
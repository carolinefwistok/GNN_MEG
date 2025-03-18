import os
import re

def delete_graph_files(processed_dir, file_index_to_delete):
    '''
    Deletes graph files in the processed directory with the specified file index.

    INPUT:
        - processed_dir: Path to the directory containing the processed graph files
        - file_index_to_delete: The file index of the graph files to delete

    OUTPUT: N/A
    '''
    # Check if the file index to delete is a list
    if not isinstance(file_index_to_delete, list):
        file_index_to_delete = [file_index_to_delete]

    for idx in file_index_to_delete:
        # Compile a regex pattern to match the graph filenames with the specified file index
        pattern = re.compile(rf'graph_{idx}_(\d+)\.pt')

        # Iterate through all files in the processed directory
        for filename in os.listdir(processed_dir):
            # Check if the filename matches the pattern
            match = pattern.match(filename)
            if match:
                # Get the full path for the file to delete
                filepath = os.path.join(processed_dir, filename)
                # Delete the file
                os.remove(filepath)
                print(f'Deleted {filename}')

def rename_graph_files(processed_dir, old_file_index, new_file_index):
    '''
    Renames graph files in the processed directory by changing the file index.

    INPUT:
        - processed_dir: Path to the directory containing the processed graph files
        - old_file_index: The current file index to be replaced
        - new_file_index: The new file index to replace the old one

    OUTPUT: N/A
    '''
    # Compile a regex pattern to match the graph filenames
    pattern = re.compile(rf'graph_{old_file_index}_(\d+)\.pt')

    # Initialize a counter for the new graph index
    new_graph_index = 0

    # Iterate through all files in the processed directory
    for filename in os.listdir(processed_dir):
        # Check if the filename matches the pattern
        match = pattern.match(filename)
        if match:
            # Extract the graph index from the filename
            graph_index = match.group(1)
            # Create the new filename with the updated file index
            new_filename = f'graph_{new_file_index}_{graph_index}.pt'
            # Get the full paths for the old and new filenames
            old_filepath = os.path.join(processed_dir, filename)
            new_filepath = os.path.join(processed_dir, new_filename)
            # Rename the file
            os.rename(old_filepath, new_filepath)
            print(f'Renamed {filename} to {new_filename}')
            # Increment the new graph index
            new_graph_index += 1

if __name__ == "__main__":
    # Define the path to the processed directory
    processed_dir = r'F:\MEG GNN\GNN\Data\Processed'

    # Define the file index to delete
    file_index_to_delete = []
    # Call the function to delete the graph files with the specified file index
    delete_graph_files(processed_dir, file_index_to_delete)

    # Define the old and new file indices
    old_file_index = 1
    new_file_index = 17

    # Call the function to rename the graph files
    rename_graph_files(processed_dir, old_file_index, new_file_index)
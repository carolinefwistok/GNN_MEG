## TM3 Graduation Project - Caroline Witstok
Graph Neural Network model for classifying stimulation status (ON/OFF) in patients with Spinal Cord Stimulation


### Installing requirements
You will need to install all requirements listed in `requirements.txt`
If you are using a virtual environment, you can install the requirements using the following command:
``` 
python -m pip install -r requirements.txt
```

Note: make sure that your virtual environment is located in the same directory as the project (and the requirements file).


### Folders
- `data`: contains the dataset used for training and testing the model
  - `processed`: contains the graphs that are created from the data in the folder `data\raw` 
  - `processed_saved`: contains folders that store previously created graphs, which can be copy pasted to the `processed` folder to save time
  - `raw`: contains the MEG data (in fif-file format) used for creating the graphs
    - `export from Brainstorm`: contains the Python script used to export the data from Brainstorm (FT data)
    - `fif files - ...`: folders containing MEG data in (fif-file format) that are currently NOT used when running the model, but act as storage

- `ray_results`: contains the results of the Ray Tune hyperparameter tuning process
- `ray_temp`: contains temporary files created during the Ray Tune process


### Scripts
- `main.py`: contains the main function that trains and tests the model (including hyperparameter tuning via Ray)
- `dataset.py`: creates the graphs from the MEG data in the `data\raw` folder
- `data_utils.py`: contains function(s) to load and preprocess the data
- `model.py`: defines the Graph Neural Network model
- `train.py`: trains the model and logs the results in MLflow
- `utils.py`: contains functions to plot and save the results
- `visualize_graphs.py`: creates a plot of the graph structure and the PSD
- `data\raw\export from Brainstorm\export_ftp_python.py`: Python script used to export the data from Brainstorm (FT data) into fif-files (compatible with MNE Python)


### Running the model
* For running the model, the script `main.py` should be used.
* Make sure that the correct fif-file data files are present in the `data\raw` folder.
* Check if there are previously created graphs for these fif files with these processing settings in the `data\processed_saved` folder.
  * And if so, copy and paste these graphs to the `data\processed` folder.
* Initialize the MLflow logging by running `mlflow ui` in the terminal.
  * This will start the MLflow UI server.
  * You can then access the UI at `http://localhost:5000` in your browser.
  * In here, the results of the model will be logged and can be viewed (only if the MLflow logging is enabled in the terminal).
* Run the `main.py` script.
  * If no stimulation information is provided in the dedicated excel sheet, you are asked to enter this information manually in the terminal in order to continue the run.


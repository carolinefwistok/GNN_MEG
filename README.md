## TM3 Graduation Project - Caroline Witstok
Graph Neural Network model for classifying stimulation status (ON/OFF) in patients with Spinal Cord Stimulation.

------------

### Installing requirements
You will need to install all requirements listed in `requirements.yaml`.
First, make sure that Anaconda is installed on your system (https://www.anaconda.com/download/success).
To create a virtual environment in conda with all required packages (`GNNenv`), use the following command in Anaconda Prompt:
``` 
conda env create --name GNNenv --file requirements.yaml
```
The environment will be created in the Anaconda directory (`~anaconda3\envs\GNNenv`), which will also hold the preferred version of Python (in this case Python 3.11)

Then activate your virtual environment using:
```
conda activate GNNenv
```

Add new packages using either conda or pip, as needed:
```
conda install <package_name>
pip install <package_name>
```

Saving these requirements as a `.yaml` file can be done using this command:
```
conda env export > <path-to-project>\requirements.yaml
```

When opening the project files in VSCode (or any other preferred code editor), make sure to select the virtual environment when running the code.
In VSCode, you can do this by pressing "ctrl"+"shift"+"P" and click on "Python: Select Interpreter" to select the `GNNenv` environment.

------------

### Directories
The project is structured as follows on the hard drive `F:MEG_connect` and on a local directory `C:\Users\carow\Documents\GNN`.

------------

The hard drive directory holds the raw data and the processed data.
- `MEG GNN`: this folder stores all files related to this graduation project.
  - `brainstorm_db`: this folder containst the Brainstorm process for preprocessing purposes.
  - `GNN`: this folder contains all GNN model related files.
    - The contents of this folder will be elaborated in the next section.
  - `MEG data`: this folder contains all raw MEG recordings of the patients with cyclic SCS.
    - This folder also contains `MEG_PT_notes.xlsx`, which lists the stimulation settings and OFF times for each file.

### Folders within 'GNN' project folder
- `data`: contains the dataset used for training and testing the model.
  - `processed`: contains the graphs that are created from the data in the folder `data\raw` .
    - `bad_subepochs`: contains the graphs based on subepochs that were removed from the dataset due to artifacts, including the graph file, text file with the subepoch indices, and a saved plot of the time series of the bad subepoch.
  - `processed_saved`: contains folders that store previously created graphs, which can be copy pasted to the `processed` folder to save time.
  - `raw`: contains the MEG data (in fif-file format) used for creating the graphs.
    - `export from Brainstorm`: contains the Python script used to export the data from Brainstorm (FT data).
    - `fif files - NOT USED`: folder containing MEG data in (fif-file format) that are currently NOT used when running the model, but act as storage.
    - `scout files`: contains scout time series in .h5 format.
      - `Mat files`: contains the .mat files that are exported from Brainstorm.
      - `scout files - NOT USED`: folder containing scout time series that are currently NOT used when running the model, but act as storage.
- `MLflow`: this folder is created for the MLflow logging.
- `output`: in here, the excel files containing the model results will be stored, the subfolders specify the epoch settings. The `Training log.xlsx` file contains a structured overview and notes of these excel files.
- `ray_results`: contains the results of the Ray Tune hyperparameter tuning process.
- `ray_temp`: contains temporary files created during the Ray Tune process.

------------

The local directory holds all Python scripts, and is used to run the model.

### Scripts
- `main.py`: contains the main function that trains and tests the model (including hyperparameter tuning via Ray).
- `dataset.py`: creates the graphs from the MEG data in the `data\raw` folder.
- `data_utils.py`: contains function(s) to load and preprocess the data.
- `model.py`: defines the Graph Neural Network model.
- `train.py`: trains the model and logs the results in MLflow.
- `utils.py`: contains functions to plot and save the results.
- `visualize_graphs.py`: creates a plot of the graph structure and the PSD.
- `export_scouts.py`: exports a source estimated scout file from Brainstorm (.mat format) to a h5py (Python) file.
- `export_ftp_python.py`: Python script used to export the data from Brainstorm (FT data) into fif-files (compatible with MNE Python).

------------

### Running the model
* For running the model, the script `main.py` should be used on the local directory of your computer.
  * Specify whether you want to train a new model (and specify the model name), or test a saved model (and specify the saved model name).
  * Inside the `create_dataset()` function, specify whether you want to run fif-files or scout files.
* Make sure that the correct fif-file data files are present in the `data\raw` folder on the hard drive (also if you are running scout time series, the corresponding fif-files should be present inside the `data\raw` folder).
* Check if there are previously created graphs for these fif files with these processing settings in the `data\processed_saved` folder on the hard drive.
  * And if so, copy and paste these graphs to the `data\processed` folder.
* Ensure that your device can handle long directory paths (for saving the hyperparamters) using https://www.howtogeek.com/266621/how-to-make-windows-10-accept-file-paths-over-260-characters/.
* Initialize the MLflow logging by navigating to the `MLFlow` folder on the hard drive (`F:MEG_connect\MEG GNN\GNN\MLflow`), open this folder in the terminal (right-click on inside the folder in File Explorer, and click "Open in Terminal") and run the command `mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000` in the terminal.
  * This will start the MLflow UI server.
  * You can then access the UI at `http://localhost:5000` in your browser.
  * In here, the results of the model will be logged and can be viewed (only if the MLflow logging is enabled in the terminal).
* Run the `main.py` script on the local directory.
  * If no stimulation information for the input data is provided in the dedicated excel sheet, you are asked to enter this information manually in the terminal in order to continue the run.


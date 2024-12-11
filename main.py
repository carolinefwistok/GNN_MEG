import numpy as np
import os
from sklearn.model_selection import train_test_split
import mlflow
import ray
from ray import train, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow
from dataset import MEGGraphs
from data_utils import get_stimulation_info
from train import train_func, test_best_model
from visualize_graphs import plot_PSD, visualize_graph
from utils import plot_train_results, save_results
import datetime


def create_dataset():
    '''
    Calls the MEGGraphs class (see dataset.py) to create a dataset of graphs out of raw MEG data. The inputs needed for the MEGGraphs class are defined here. 
    INPUT: N/A
    OUTPUT: 
        - dataset       : Dataset of graphs
        - duration      : Duration of subepochs in seconds
        - overlap       : Overlap of subepochs in seconds
        - num_graphs    : Number of graphs in the dataset
        - fmax          : Maximum frequency of the computed PSD 

    '''

    # Define filenames of MEG data exported as fif file
    directory = r'F:\MEG GNN\GNN\Data\Raw'
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith('.fif'):
            filenames.append(filename)

    # Sort filenames alphabetically
    filenames = sorted(filenames)

    print('filenames', filenames)

    # Define path to stimulation information Excel file
    stim_excel_file = r'F:\MEG GNN\MEG data\MEG_PT_notes.xlsx'

    # Retrieve stimulation info for each file and store it in a dictionary
    stim_info_dict = {}
    for filename in filenames:
        stim_info = get_stimulation_info(stim_excel_file, filename)
        
        if stim_info is not None:
            print(f"Stimulation info for {filename}:")
            print(stim_info)
        else:
            print(f"No stimulation info found or enterred for {filename}")

        stim_info_dict[filename] = stim_info

    # Define the duration of the subepochs that are created out of the epochs and the amount of overlap between them (in seconds)
    duration = 30
    overlap = 25

    # Define the ramp time (in seconds) for the subepochs to account for ramping of stimulation effects
    ramp_time = 5

    # Define maximum frequency for PSD calculation (maximum is resample_freq/2 due to Nyquist theorem)
    fmax = 128

    # Define connectivity method for defining edges
    conn_method = 'pli'
    freqs = {'fmin': 1, 
            'fmax': 40, 
            'freqs': np.linspace(1, 40, (40 - 1) * 4 + 1)}
    
    # Define root directory to the 'raw' and 'processed' folders
    root_directory = r'F:\MEG GNN\GNN\Data'
    
    # Call the MEGGraphs class (see dataset.py)
    dataset = MEGGraphs(root=root_directory,
                        filenames=filenames,
                        stim_info_dict=stim_info_dict, 
                        duration=duration,
                        overlap=overlap,
                        conn_method=conn_method,
                        freqs=freqs,
                        fmax=fmax,
                        ramp_time=ramp_time)
    
    # Retrieve total number of graphs in the dataset
    num_graphs = len(dataset)
    
    return dataset, duration, overlap, num_graphs, fmax

def split_train_test(dataset):
    '''
    Splits the Dataset object into a train and test set. 
    INPUT:
        - dataset           : Dataset of graphs
    
    OUTPUTS: 
        - dataset_train     : Dataset of graphs for training
        - dataset_test      : Dataset of graphs for testing 
        - y_train           : list of labels of train set 
        - y_test            : list of labels of test set

    '''
    # Retrieve the labels from all the graphs in the dataset object
    labels = [data.y for data in dataset]

    # Perform the stratified train test split
    dataset_train, dataset_test, y_train, y_test = train_test_split(
        dataset,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    return dataset_train, dataset_test, y_train, y_test

def train_hyperparameters(mlflow_tracking_uri, dataset, dataset_train, y_train):
    '''
    Trains the GNN model (see model.py) using the Ray trainable train_func (see train.py). The search space used for the hyperparameter tuning is defined here.
    INPUTS: 
        - dataset           : Dataset of graphs
        - dataset_train     : Dataset of graphs for training 
        - y_train           : list of labels of train set 
    
    OUTPUTS: 
        - results           : ResultGrid object of results of all hyperparameter configurations
        - best_result       : Result object of results of best hyperparameter configuration
        - best_params       : dictionary of the best hyperparameter configuration
    '''

    # Terminate processes started by ray.init(), so you can define a local _temp_dir to store the Ray process files
    if ray.is_initialized():
        ray.shutdown()
    ray.init(_temp_dir=r"F:\MEG GNN\GNN\Ray_temp")

    # Initialize MLflow logging
    experiment_name = f"mlflow_hyperparameter_tuning_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Make sure Ray doesn't change the working directory to the trial directory, so you can define your own (relative) path to store results 
    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"

    # Make sure Ray can handle reporting more than one metric 
    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

    # Define hyperparameter search space 
    search_space = {
        'hidden_channels': tune.choice([16, 32, 64, 128]),
        'lr': tune.loguniform(1e-5, 1e-2),
        'batch_size': tune.choice([2, 4, 6, 8])
    }

    # Define path where results need to be stored
    run_config = train.RunConfig(
        name = 'tune_hyperparameters',
        storage_path=r"F:\MEG GNN\GNN\Ray_results",
        callbacks=[
            MLflowLoggerCallback(
                tracking_uri=mlflow_tracking_uri,
                experiment_name=experiment_name,
                save_artifact=True,
            )
        ],
    )               

    # Define which metric you want Ray to base 'best_results' on, whether that metric needs to be 'max' or 'min', and how many configurations you want it to try
    tune_config = tune.TuneConfig(
        metric='val_accuracy',
        mode='max',
        num_samples=15
    )

    # Perform the training and hyperparameter tuning
    tuner = tune.Tuner(
        tune.with_parameters(train_func, dataset=dataset, dataset_train=dataset_train, y_train=y_train),
        param_space=search_space,
        tune_config=tune_config,
        run_config=run_config
    )

    results = tuner.fit()

    # Retrieve best result
    best_result = results.get_best_result()

    # Retrieve hyperparameter configuration of best result
    best_params = best_result.config
    return results, best_result, best_params

def main():
    '''
    Calls all functions in this script. 
    INPUT: N/A
    OUTPUT: N/A
    '''
    # Create  dataset of graphs
    dataset, duration, overlap, num_graphs, fmax = create_dataset()

    # If plotting = True, plot and visualize the dataset
    plotting = False
    if plotting:
        plot_PSD(dataset, fmax)
        visualize_graph(dataset)

    # Train the model on the training set
    dataset_train, dataset_test, y_train, y_test = split_train_test(dataset)
    print('training set', len(dataset_train))
    print('test set', len(dataset_test))
    
    # Specify tracking server
    mlflow_tracking_uri = "http://localhost:5000"
    results, best_result, best_params = train_hyperparameters(mlflow_tracking_uri, dataset, dataset_train, y_train)

    # If plotting = True, plot the results of the model
    if plotting:
        plot_train_results(results, best_result)

    # Save model results and best parameters
    results_df = results.get_dataframe()
    output_directory = r'F:\MEG GNN\GNN\Output'
    save_results(results_df, best_params, best_result, output_directory, duration, overlap, num_graphs)

    # Print test accuracy of best performing model
    acc_test = test_best_model(best_result, dataset, dataset_test)
    print(f'Test accuracy: {acc_test}')

if __name__ == "__main__":
    '''
    Runs the main function. 
    '''
    main()

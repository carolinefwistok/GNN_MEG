import numpy as np
import os
from sklearn.model_selection import train_test_split
import mlflow
import logging
import torch
from datetime import datetime
import ray
from ray import train, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow
from ray.tune.schedulers import AsyncHyperBandScheduler
import multiprocessing

from dataset import MEGGraphs
from data_utils import *
from train import *
from visualize_graphs import *
from utils import *


# Specificy the state of this run (training and saving a new model or testing, using a saved model)
run_state = 'training'  # 'training' or 'testing'

# Specifiy the model name
model_name = '_PTN0406_BURST_TONIC_baseline_PSD_negatives_30dur_25overl' 

# Specify whether plots should be generated when running (set to True), or not (set to False)
generate_plots = False


def create_dataset():
    '''
    Calls the MEGGraphs class (see dataset.py) to create a dataset of graphs out of raw MEG data.
    The inputs needed for the MEGGraphs class are defined here.

    INPUT: N/A
    OUTPUT: 
        - dataset       : Dataset of graphs
        - duration      : Duration of subepochs in seconds
        - overlap       : Overlap of subepochs in seconds
        - num_graphs    : Number of graphs in the dataset
        - fmax          : Maximum frequency of the computed PSD
    '''

    # Specify the type of input data ('fif' or 'scout')
    input_type = 'fif'

    if input_type == 'fif':
        # Define filenames of MEG data exported as fif file
        directory = r'F:\MEG GNN\GNN\Data\Raw'
        filenames = []
        for filename in os.listdir(directory):
            if filename.endswith('.fif'):
                filenames.append(filename)

        # Define scouts data list as None
        scouts_data_list = None

    elif input_type == 'scout':
        # Define filenames of MEG data exported as scouts file
        directory = r'F:\MEG GNN\GNN\Data\Raw\Scout files'
        fif_directory = r'F:\MEG GNN\GNN\Data\Raw'

        scout_filenames = []
        for scout_filename in os.listdir(directory):
            if scout_filename.endswith('.h5'):
                scout_filenames.append(scout_filename)
        
        # Sort scout filenames alphabetically
        scout_filenames = sorted(scout_filenames)
        print('scout filenames', scout_filenames)

        # Retrieve list of dictionaries containing the scouts data for each file
        scouts_data_list = load_scouts_data(scout_filenames, directory)

        # Retrieve list of raw fif filenames corresponding to the scout data files
        filenames = get_raw_file(scout_filenames, fif_directory)
    else:
        print(f'Input type {input_type} not recognized')

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
            print(f"Stimulation info found for {filename}:")
        else:
            print(f"No stimulation info found or enterred for {filename}")
        stim_info_dict[filename] = stim_info

    # Define the duration of the subepochs that are created out of the epochs and the amount of overlap between them (in seconds)
    duration = 30
    overlap = 25

    # Define the ramp time (in seconds) for the subepochs to account for ramping of stimulation effects
    ramp_time = 5

    # Define minimum and maximum frequency for PSD calculation (maximum is resample_freq/2 due to Nyquist theorem; 128 Hz in this case)
    fmin = 1
    fmax = 128

    # Define connectivity method for defining edges
    conn_method = 'pli'
    freqs = {'fmin': fmin,
            'fmax': fmax,
            'freqs': np.linspace(1, fmax, (fmax - 1) * 4 + 1)}

    # Define root directory to the 'raw' and 'processed' folders that store the MEG data and the graphs, respectively
    root_directory = r'F:\MEG GNN\GNN\Data'
    
    # Call the MEGGraphs class (see dataset.py)
    if input_type == 'fif':
        dataset = MEGGraphs(input_type=input_type,
                            root=root_directory,
                            filenames=filenames,
                            stim_info_dict=stim_info_dict, 
                            duration=duration,
                            overlap=overlap,
                            conn_method=conn_method,
                            freqs=freqs,
                            fmin=fmin,
                            fmax=fmax,
                            ramp_time=ramp_time
        )
    # If the input is scout time series data, add the scouts_data_list argument
    elif input_type == 'scout':
        dataset = MEGGraphs(input_type=input_type,
                            root=root_directory,
                            filenames=filenames,
                            stim_info_dict=stim_info_dict, 
                            duration=duration,
                            overlap=overlap,
                            conn_method=conn_method,
                            freqs=freqs,
                            fmin=fmin,
                            fmax=fmax,
                            ramp_time=ramp_time,
                            scout_data_list=scouts_data_list  # Only added if input_type is 'scout'
        )

    print('dataset', dataset)
    print('total graphs per file', dataset.graphs_per_file())
    print('total stim ON graphs per file', dataset.stim_graphs_per_file())
    print('total stim OFF graphs per file', dataset.non_stim_graphs_per_file())

    # Retrieve total number of graphs in the dataset
    num_graphs = len(dataset)

    return dataset, duration, overlap, num_graphs, fmax, scouts_data_list

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
    labels = [data.y.item() for data in dataset if data.y is not None]

    # Perform the stratified train test split
    dataset_train, dataset_test, y_train, y_test = train_test_split(
        dataset,
        labels,
        test_size=0.2,
        random_state=1,
        stratify=labels
    )

    return dataset_train, dataset_test, y_train, y_test

def short_trial_dirname_creator(trial):
    '''
    Create a short name for the trial based on its parameters. This is used to define the directory where the results
    of the hyperparameter tuning are stored.

    INPUTS:
        - trial     : Trial object

    OUTPUTS:
        - string    : Short name for the trial
    '''

    params = trial.config
    return f"trial_{trial.trial_id[:8]}_n_layers={params['n_layers']}_lr={params['lr']}_bs={params['batch_size']}"

def train_hyperparameters(dataset, dataset_train, y_train, model_name=None):
    '''
    Trains the GNN model (see model.py) using the Ray trainable train_func (see train.py).
    The search space used for the hyperparameter tuning is defined here.

    INPUTS: 
        - dataset           : Dataset of graphs
        - dataset_train     : Dataset of graphs for training 
        - y_train           : list of labels of train set
        - model_name        : Name of the model to be saved (if None, a new model will be trained)
    
    OUTPUTS: 
        - results           : ResultGrid object of results of all hyperparameter configurations
        - best_result       : Result object of results of best hyperparameter configuration
        - best_params       : dictionary of the best hyperparameter configuration
    '''

    print('Setting up MLflow URI...')
    # Set the MLflow tracking URI
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    
    print('Initializing Ray...')
    # Terminate processes started by ray.init(), so you can define a local _temp_dir to store the Ray process files
    if ray.is_initialized():
        ray.shutdown()
        print('Ray shutdown.')

    # Disable the Ray dashboard
    # os.environ["RAY_DISABLE_DASHBOARD"] = "1"
 
    ray.init(_temp_dir=r"F:\MEG GNN\GNN\Ray_temp", logging_level=logging.ERROR)
    print('Ray initialized.')

    print('Initilizing MLflow logging...')
    # Initialize MLflow logging
    experiment_name = f"mlflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(experiment_name)
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    print('MLflow tracking URI:', mlflow_tracking_uri)

    # Make sure Ray doesn't change the working directory to the trial directory, so you can define your own (relative) path to store results 
    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"

    # Make sure Ray can handle reporting more than one metric 
    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

    # Define hyperparameter search space
    search_space = {
        'n_layers': tune.choice([2, 3, 4]),
        'dropout_rate': tune.choice([0.01, 0.1, 0.3, 0.5]),
        # 'dropout_rate': tune.uniform(0.3, 0.9),
        'conv1_hidden_channels': tune.choice([16, 32, 64, 128]),
        # 'conv2_hidden_channels': tune.choice([16, 32, 64, 128]),
        # 'conv3_hidden_channels': tune.choice([16, 32, 64, 128]),
        # 'conv4_hidden_channels': tune.choice([16, 32, 64, 128]),
        'lr': tune.choice([0.00001, 0.0001, 0.001, 0.01]),
        # 'lr': tune.loguniform(1e-1, 1e-5),
        'batch_size': tune.choice([2, 4, 8, 16, 32, 64, 128]),
        'weight_decay': tune.choice([0.00001, 0.0001, 0.001]),
        'top_k': tune.choice([None, 300, 600, 900, 1200]),
        'threshold': tune.choice([None, 0.01, 0.03, 0.05, 0.07])
    }

    # Define the scheduler for hyperparameter tuning
    scheduler = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        max_t=100,
        grace_period=25,
        reduction_factor=6
    )

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

    # Define how Ray should choose the 'best_results'
    tune_config = tune.TuneConfig(
        metric='val_accuracy',
        mode='max',
        num_samples=20,
        scheduler=scheduler,
        trial_dirname_creator=short_trial_dirname_creator,
    )

    # Perform the training and hyperparameter tuning
    tuner = tune.Tuner(
        tune.with_parameters(train_func, dataset=dataset, dataset_train=dataset_train, y_train=y_train, model_name=model_name),
        param_space=search_space,
        tune_config=tune_config,
        run_config=run_config
    )
    results = tuner.fit()

    # Retrieve best result
    best_result = results.get_best_result()

    # Retrieve hyperparameter configuration of best result
    best_params = best_result.config

    # Save the best result and hyperparameters
    output_dir = os.path.join(os.path.dirname(__file__), 'Output')
    os.makedirs(output_dir, exist_ok=True)
    best_result_path = os.path.join(output_dir, f"best_result{model_name}.pt")
    torch.save({
        'best_result': best_result,
        'best_params': best_params
    }, best_result_path)

    return results, best_result, best_params

def load_best_result(best_result_path):
    '''
    Loads the saved best result and its configuration parameters.

    INPUT:
        - best_result_path : Path to the saved best result file
    
    OUTPUT:
        - best_result      : Loaded best result object
        - best_params      : Loaded configuration parameters
    '''

    checkpoint = torch.load(best_result_path, weights_only=False)
    best_result = checkpoint['best_result']
    best_params = checkpoint['best_params']
    print('best result', best_result)
    print('best params', best_params)

    return best_result, best_params

def filter_dataset(dataset_train, dataset_test, best_result):
    '''
    Apply edge filtering using TopK filtering and threshold filtering to the train and test dataset with
    the best performing hyperparameters.

    INPUTS:
        - dataset_train             : Dataset list of graphs for training
        - dataset_test              : Dataset list of graphs for testing
        - best_result               : Result object of results of best hyperparameter configuration

    OUTPUTS:
        - filtered_dataset_train    : Dataset list of graphs for training with filtered edges
        - filtered_dataset_test     : Dataset list of graphs for testing with filtered edges
    '''

    # Retrieve hyperparameter configuration for the TopK and threshold values of the best performing result
    top_k = best_result.config['top_k']
    print('top k', top_k)
    threshold = best_result.config['threshold']
    print('threshold', threshold)

    # Apply edge filtering to the train and test dataset
    filtered_dataset_train = dataset_train.copy()
    filtered_dataset_train = [edge_filtering(graph, top_k, threshold) for graph in filtered_dataset_train]
    filtered_dataset_test = dataset_test.copy()
    filtered_dataset_test = [edge_filtering(graph, top_k, threshold) for graph in filtered_dataset_test]

    return filtered_dataset_train, filtered_dataset_test

def filter_entire_dataset(dataset, best_result):
    '''
    Apply edge filtering using TopK filtering and threshold filtering to the entire dataset with
    the best performing hyperparameters.

    INPUTS:
        - dataset                   : Dataset of graphs
        - best_result               : Result object of results of best hyperparameter configuration

    OUTPUTS:
        - filtered_dataset_complete : Dataset of graphs for testing a saved model (not split in train and test set)
    '''

    # Retrieve hyperparameter configuration for the TopK and threshold values of the best performing result
    top_k = best_result.config['top_k']
    print('top k', top_k)
    threshold = best_result.config['threshold']
    print('threshold', threshold)

    # Convert dataset to a list of graphs
    graph_list = [dataset[i] for i in range(len(dataset))]

    # Apply edge filtering to the entire dataset
    filtered_dataset_complete = [edge_filtering(graph, top_k, threshold) for graph in graph_list]

    return filtered_dataset_complete

def main(run_state, model_name, generate_plots):
    '''
    Calls all functions in this script.

    INPUT: N/A
    OUTPUT: N/A
    '''

    # Create dataset of graphs
    dataset, duration, overlap, num_graphs, fmax, scouts_data_list = create_dataset()

    # Remove bad graphs from the dataset after the dataset has been created
    dataset.remove_bad_graphs()
    print('dataset', len(dataset))

    # Check the state of the run (training or testing) and specify a path to save or load the model
    if run_state == 'training':
        # If a new model is trained, the model path is None, and a new model will be saved to the Output folder
        model_path = None
    elif run_state == 'testing':
        # If a saved model is tested, the model path is specified
        model_path = os.path.join(os.path.dirname(__file__), 'Output', f'best_model{model_name}.pt')
    else:
        print(f'State {run_state} is not configured correctly to either training or testing.')
    print('Model path:', model_path)

    # If generate_plots = True, plot and visualize the dataset
    if generate_plots:
        if scouts_data_list is None:
            # Plot PSDs of subepochs
            # plot_PSD_avg_per_file(dataset, fmax)
            plot_PSD_avg(dataset, fmax)

            # Visualize graph with nodes and edges
            visualize_graph(dataset)

            # Plot connectivity matrix of the first graph
            plot_connectivity_matrix(dataset, file_index=0)
        else:
            # Plot PSDs of subepochs
            # plot_PSD_avg_per_file(dataset, fmax)
            plot_PSD_avg_scouts(dataset, scouts_data_list, fmax)

            # Visualize graph with scouts as nodes
            visualize_scout_graph(dataset, scouts_data_list)

            # Plot connectivity matrix of the first graph with scout data
            plot_scout_connectivity_matrix(dataset, scouts_data_list, file_index=0)

    if model_path is None:
        # Train the model on the training set
        dataset_train, dataset_test, y_train, y_test = split_train_test(dataset)
        print('training set', len(dataset_train))
        print('test set', len(dataset_test))
        results, best_result, best_params = train_hyperparameters(dataset, dataset_train, y_train, model_name)

        # Filter the dataset using the best hyperparameters for edge filtering
        filtered_dataset_train, filtered_dataset_test = filter_dataset(dataset_train, dataset_test, best_result)

        # If generate_plots = True, plot and visualize the filtered dataset
        generate_plots = True
        if generate_plots:
            if scouts_data_list is None:
                visualize_filtered_graph(filtered_dataset_train[0])
            else:
                visualize_filtered_graph(filtered_dataset_train[0], input_type='scout', scouts_data_list=scouts_data_list)

        # If generate_plots = True, plot the results of the model
        if generate_plots:
            plot_train_results(results, best_result)

        # Save model results and best parameters
        results_df = results.get_dataframe()
        output_directory = r'F:\MEG GNN\GNN\Output'
        save_results(results_df, best_params, best_result, output_directory, duration, overlap, num_graphs)

        # Print test accuracy of best performing model
        acc_test, roc_auc, precision, recall, conf_matrix = test_best_model(best_result, dataset, filtered_dataset_test)
        print(f'Test accuracy: {acc_test}')

        # Save metrics to text file
        metrics = {
            "Test Accuracy": acc_test,
            "ROC AUC": roc_auc,
            "Precision": precision,
            "Recall": recall,
            "Confusion Matrix": conf_matrix.tolist()
        }
        save_metrics_to_txt(metrics, dataset.get_filenames(), output_directory, f"train_test{model_name}.txt")

    elif model_path is not None:
        print('Loading saved model...')
        # Load the saved best result and hyperparameters
        best_result_path = os.path.join(os.path.dirname(__file__), 'Output', f'best_result{model_name}.pt')
        best_result, best_params = load_best_result(best_result_path)
        print('Best result:', best_result)
        print('Best params:', best_params)

        # Filter the entire dataset using the best hyperparameters for edge filtering
        filtered_dataset_complete = filter_entire_dataset(dataset, best_result)

        # Load and test a saved model on the current dataset
        acc_test, roc_auc, precision, recall, conf_matrix = load_and_test_model(model_path, dataset, filtered_dataset_complete)
        print(f'Test accuracy: {acc_test}')

        # Save metrics to text file
        metrics = {
            "Test Accuracy": acc_test,
            "ROC AUC": roc_auc,
            "Precision": precision,
            "Recall": recall,
            "Confusion Matrix": conf_matrix.tolist()
        }
        output_directory = r'F:\MEG GNN\GNN\Output'
        save_metrics_to_txt(metrics, dataset.get_filenames(), output_directory, f"test_{model_name}.txt")

if __name__ == "__main__":
    '''
    Runs the main function. 
    '''
    
    multiprocessing.set_start_method('spawn', force=True)
    main(run_state, model_name, generate_plots)

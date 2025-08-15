import torch
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score, auc
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import GradScaler, autocast
import ray
from ray.tune import get_checkpoint, report, Checkpoint, RunConfig
from ray import tune
# from ray import train, tune
from ray.tune import RunConfig
from ray.tune.callback import Callback
# from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow
from ray.tune.schedulers import AsyncHyperBandScheduler
import tempfile
import json
import mlflow.pytorch
import torch.nn.functional as F
from dataset import MEGGraphs
from model import GNN
from utils import edge_filtering, mst_filtering
from explain import *
import itertools

class ProgressCallback(Callback):
    def __init__(self, total_trials):
        self.total_trials = total_trials
        self.start_time = time.time()
        self.completed_trials = 0
        
    def on_trial_complete(self, iteration, trials, trial, **info):
        self.completed_trials += 1
        elapsed_time = time.time() - self.start_time
        
        # Calculate progress and time estimates
        progress_pct = (self.completed_trials / self.total_trials) * 100
        avg_time_per_trial = elapsed_time / self.completed_trials
        estimated_total_time = avg_time_per_trial * self.total_trials
        estimated_remaining_time = estimated_total_time - elapsed_time
        
        # Format time strings
        elapsed_str = self._format_time(elapsed_time)
        remaining_str = self._format_time(estimated_remaining_time)
        total_str = self._format_time(estimated_total_time)
        
        print(f"\n{'='*60}")
        print(f"HYPERPARAMETER OPTIMIZATION PROGRESS")
        print(f"{'='*60}")
        print(f"Completed: {self.completed_trials}/{self.total_trials} trials ({progress_pct:.1f}%)")
        print(f"Elapsed time: {elapsed_str}")
        print(f"Estimated remaining: {remaining_str}")
        print(f"Estimated total time: {total_str}")
        print(f"Average time per trial: {self._format_time(avg_time_per_trial)}")
        
        # Show best result so far
        if hasattr(trial, 'last_result'):
            best_val_acc = trial.last_result.get('val_accuracy', 0)
            print(f"Current trial val_accuracy: {best_val_acc:.4f}")
        
        print(f"{'='*60}\n")
    
    def _format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

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
    
    print('Initializing Ray...')
    # Terminate processes started by ray.init(), so you can define a local _temp_dir to store the Ray process files
    if ray.is_initialized():
        print('Ray is already initialized, shutting down...')
        ray.shutdown()
        print('Ray shutdown.')

    # Make sure Ray doesn't change the working directory to the trial directory, so you can define your own (relative) path to store results 
    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"

    # Make sure Ray can handle reporting more than one metric 
    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

    print('Starting Ray locally...')
    os.makedirs("/scratch/cwitstok/tmp/ray_tmp", exist_ok=True)
    print(f"Ray temp dir exists? {os.path.exists('/scratch/cwitstok/tmp/ray_tmp')}")
    print(f"Writable? {os.access('/scratch/cwitstok/tmp/ray_tmp', os.W_OK)}")
    ray.init(_temp_dir="/scratch/cwitstok/tmp/ray_tmp", num_cpus=10, logging_level="ERROR", local_mode=True)
    print('Ray initialized.')

    # Define hyperparameter search space
    search_space = {
        'n_layers': tune.choice([2, 3, 4]),
        'dropout_rate': tune.choice([0.01, 0.1, 0.3, 0.5]),
        'conv1_hidden_channels': tune.choice([16, 32, 64, 128]),
        'lr': tune.choice([0.00001, 0.0001, 0.001, 0.01]),
        'batch_size': tune.choice([2, 4, 8, 16, 32, 64, 128]),
        'weight_decay': tune.choice([0.00001, 0.0001, 0.001]),
        'top_k': tune.choice([None, 300, 600, 900, 1200]),
        'threshold': tune.choice([None, 0.01, 0.03, 0.05, 0.07]),
        'edge_filtering': tune.choice(['Threshold_TopK', 'MST'])
    }

    # Define the number of samples to be drawn from the search space
    num_samples = 30

    # Create progress callback
    progress_callback = ProgressCallback(num_samples)

    # Define the scheduler for hyperparameter tuning
    scheduler = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        max_t=100,
        grace_period=25,
        reduction_factor=3
    )

    # Define path where results need to be stored
    run_config = RunConfig(
        name = 'tune_hyperparameters',
        storage_path="/scratch/cwitstok/Ray_temp",
        callbacks=[progress_callback],
    )

    # Define how Ray should choose the 'best_results'
    tune_config = tune.TuneConfig(
        metric='val_accuracy',
        mode='max',
        num_samples=num_samples,
        scheduler=scheduler,
        trial_dirname_creator=short_trial_dirname_creator,
    )

    print(f"\n{'='*60}")
    print(f"STARTING HYPERPARAMETER OPTIMIZATION")
    print(f"{'='*60}")
    print(f"Total trials to run: {num_samples}")
    print(f"Search space size: {len(search_space)} parameters")
    print(f"Model: {model_name or 'default'}")
    print(f"Training samples: {len(dataset_train)}")
    print(f"{'='*60}\n")

    # Perform the training and hyperparameter tuning
    tuner = tune.Tuner(
        tune.with_parameters(train_func, dataset=dataset, dataset_train=dataset_train, y_train=y_train, model_name=model_name),
        param_space=search_space,
        tune_config=tune_config,
        run_config=run_config
    )
    
    # Start timing
    start_time = time.time()
    results = tuner.fit()
    end_time = time.time()

    # Print final summary
    total_time = end_time - start_time
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER OPTIMIZATION COMPLETED")
    print(f"{'='*60}")
    print(f"Total time: {progress_callback._format_time(total_time)}")
    print(f"Total trials completed: {len(results)}")
    print(f"Average time per trial: {progress_callback._format_time(total_time / len(results))}")
    
    # Retrieve best result and hyperparameters
    best_result = results.get_best_result()
    best_params = best_result.config

    print(f"Best validation accuracy: {best_result.metrics.get('val_accuracy', 0):.4f}")
    print(f"Best parameters: {best_params}")
    print(f"{'='*60}\n")

    # Collect best metrics
    best_metrics = {
        "train_accuracy": best_result.metrics.get("train_accuracy"),
        "val_accuracy": best_result.metrics.get("val_accuracy"),
        "train_loss": best_result.metrics.get("train_loss"),
        "val_loss": best_result.metrics.get("val_loss"),
        "training_iteration": best_result.metrics.get("training_iteration"),
    }

    # Save best hyperparameters and metrics to a JSON file
    output_dir = os.path.join(os.path.dirname(__file__), 'Output', model_name)
    os.makedirs(output_dir, exist_ok=True)
    best_result_json_path = os.path.join(output_dir, f"best_result_{model_name or 'default'}.json")
    with open(best_result_json_path, "w") as f:
        json.dump({
            "best_params": best_params,
            "best_metrics": best_metrics
        }, f, indent=4)

    # Save the best result and hyperparameters
    os.makedirs(output_dir, exist_ok=True)
    best_result_path = os.path.join(output_dir, f"best_result{model_name}.pt")
    torch.save({
        'best_result': best_result,
        'best_params': best_params
    }, best_result_path)

    return results, best_result, best_params

def train_val_split(filtered_graphs_train, y_train, grouped=True):
    '''
    Splits the training dataset into a training and validation set using either StratifiedGroupKFold (grouped)
    or StratifiedShuffleSplit (non-grouped).
    INPUTS:
        - filtered_graphs_train : List of filtered graph objects for training
        - y_train               : List of labels for the training set
        - grouped               : If True, use patient_ids for grouping; if False, ignore grouping
    OUTPUTS:
        - subset_train          : List of graph objects for the training subset
        - subset_val            : List of graph objects for the validation subset
        - y_subset_train        : List of labels for the training subset
        - y_subset_val          : List of labels for the validation subset
    '''
    labels = [data.y.item() for data in filtered_graphs_train]
    if grouped:
        patient_ids = [data.patient_id for data in filtered_graphs_train]
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, val_idx = next(sgkf.split(np.zeros(len(labels)), labels, groups=patient_ids))
    else:
        # Use StratifiedGroupKFold to ensure no patient leakage
        # Calculate number of splits based on test_size
        n_splits = max(2, int(1 / 0.2))  # e.g., test_size=0.2 -> n_splits=5
        patient_ids = [data.patient_id for data in filtered_graphs_train]
        
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Get the first split (this gives us approximately the desired test_size)
        train_idx, val_idx = next(sgkf.split(np.zeros(len(labels)), labels, groups=patient_ids))
        
        # Print split information
        unique_patients_train = len(set([patient_ids[i] for i in train_idx]))
        unique_patients_val = len(set([patient_ids[i] for i in val_idx]))

        print(f"Split info:")
        print(f"  Training: {len(train_idx)} samples from {unique_patients_train} patients")
        print(f"  Validation: {len(val_idx)} samples from {unique_patients_val} patients")
        print(f"  Actual validation size: {len(val_idx) / len(labels):.2%}")
        
        # Verify no patient leakage
        train_patients = set([patient_ids[i] for i in train_idx])
        val_patients = set([patient_ids[i] for i in val_idx])
        
        if train_patients.intersection(val_patients):
            overlapping_patients = train_patients.intersection(val_patients)
            print(f"ERROR: Patient leakage detected! Overlapping patients: {overlapping_patients}")
        else:
            print("✓ No patient leakage detected")

    subset_train = [filtered_graphs_train[i] for i in train_idx]
    subset_val = [filtered_graphs_train[i] for i in val_idx]
    y_subset_train = [labels[i] for i in train_idx]
    y_subset_val = [labels[i] for i in val_idx]

    return subset_train, subset_val, y_subset_train, y_subset_val

def train_func(config, dataset, dataset_train, y_train, model_name):   
    '''
    Ray trainable that performs the training process for each of the hyperparameter configurations.

    INPUTS: 
        - config            : Dictionary of hyperparameters
        - dataset           : Dataset of graphs
        - dataset_train     : Dataset of graphs for training 
        - y_train           : list of labels of train set 
        - model_name        : Name of the model to be saved
    OUTPUT: N/A
    ''' 

    # Apply edge filtering to the input graphs of the train set, based on the specified edge filtering method
    if config['edge_filtering'] == 'Threshold_TopK':
        # Apply edge filtering based on the top_k and threshold parameters
        top_k = config['top_k']
        threshold = config['threshold']
        filtered_graphs_train = dataset_train.copy()
        filtered_graphs_train = [edge_filtering(graph, top_k, threshold) for graph in filtered_graphs_train]
        print(f'Edge filtering applied with top_k={top_k} and threshold={threshold}')
        print(f'Graph size after filtering: {[graph.num_edges for graph in filtered_graphs_train]}')
    elif config['edge_filtering'] == 'MST':
        # Apply edge filtering based on the Minimum Spanning Tree (MST) method
        filtered_graphs_train = dataset_train.copy()
        filtered_graphs_train = [mst_filtering(graph) for graph in filtered_graphs_train]
        print('Edge filtering applied using Minimum Spanning Tree (MST)')
        print(f'Graph size after filtering: {[graph.num_edges for graph in filtered_graphs_train]}')

    # Calculate class weights
    y_train_np = np.array(y_train)
    class_counts = torch.tensor([len(y_train_np[y_train_np == 0]), len(y_train_np[y_train_np == 1])], dtype=torch.float)
    print('class counts', class_counts)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()

    # Define criterion for model training with class weights
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Retrieve GNN model (see model.py)
    model = GNN(
        n_layers=config['n_layers'],
        dropout_rate=config['dropout_rate'],
        conv1_hidden_channels=config['conv1_hidden_channels'], 
        conv2_hidden_channels=config['conv1_hidden_channels'],
        conv3_hidden_channels=config['conv1_hidden_channels'],
        conv4_hidden_channels=config['conv1_hidden_channels'],
        dataset=dataset,
    )

    # Split train data in train and validation set 
    subset_train, subset_val, y_subset_train, y_subset_val = train_val_split(filtered_graphs_train, y_train, grouped=False)  # or grouped=False
    # subset_train, subset_val, y_subset_train, y_subset_val = train_test_split(
    #     filtered_graphs_train, 
    #     y_train, 
    #     test_size=0.2, 
    #     random_state=42, 
    #     stratify=y_train
    # )

    # Batch train and validation set
    train_loader = DataLoader(
        subset_train,
        config['batch_size'],
        shuffle=True,
    )
    val_loader = DataLoader(
        subset_val,
        config['batch_size'],
        shuffle=True,
    )

    # Define number of epochs
    num_epochs = 100

    # Define optimizer and criterion with weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    #optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # Add learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 
    #     mode='min', 
    #     factor=0.5, 
    #     patience=8,  # Reduce LR after 8 epochs without improvement
    #     verbose=True, 
    #     min_lr=1e-6
    # )

    # Early stopping parameters
    best_val_loss = float('inf')  # Initialize best validation loss to infinity
    patience = 10  # Number of epochs to wait before stopping
    patience_counter = 0  # Initialize patience counter
    min_epochs = 30  # Minimum number of epochs to train

    # Define a checkpoint so the updated model parameters can be retrieved at a later moment
    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
            model.load_state_dict(checkpoint_dict["model_state"])

    # Perform training
    for epoch in range(num_epochs):
        # Put model in training mode
        model.train()

        # Iterate over each batch in train set
        train_loss = 0
        for data in train_loader:
            # Pass batch through model to obtain prediction (logit)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)

            # Measure discrepancy between prediction (out) and label (data.y)
            loss = criterion(out, data.y)

            # Compute gradient of loss
            loss.backward()

            # Update the model's parameters using the computed gradients
            optimizer.step()

            # Clear all gradients before the next iteration
            optimizer.zero_grad()

            # Add to loss 
            train_loss += loss.item()

        # Put model in evaluation mode
        model.eval()

        # Calculate accuracy of trained model on train set
        correct_train = 0
        y_train_pred = []
        y_train_true = []
        for data in train_loader:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            pred = out.argmax(dim=1)
            correct_train += int((pred == data.y).sum())
            y_train_pred.append(pred.cpu().numpy())
            y_train_true.append(data.y.cpu().numpy())

        train_acc = correct_train / len(train_loader.dataset)
        y_train_pred = np.concatenate(y_train_pred)
        y_train_true = np.concatenate(y_train_true)

        # Calculate accuracy and loss of trained model on validation set
        correct_val = 0
        val_loss = 0
        y_val_pred = []
        y_val_true = []
        for data in val_loader:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out, data.y)
            pred = out.argmax(dim=1)
            val_loss += loss.item()
            correct_val += int((pred == data.y).sum())
            y_val_pred.append(pred.cpu().numpy())
            y_val_true.append(data.y.cpu().numpy())

        val_acc = correct_val / len(val_loader.dataset)
        y_val_pred = np.concatenate(y_val_pred)
        y_val_true = np.concatenate(y_val_true)

        # Step the scheduler
        #scheduler.step(val_loss)
        # current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}") # - LR: {current_lr:.6f}")

        # Save accuracies and losses together with the checkpoint
        with tempfile.TemporaryDirectory() as tempdir:
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict()},
                os.path.join(tempdir, "checkpoint.pt"),
            )

            report(
                {"train_accuracy": train_acc, "val_accuracy": val_acc, "train_loss": train_loss, "val_loss": val_loss, "training_iteration": epoch},
                checkpoint=Checkpoint.from_directory(tempdir)
            )

        # Early stopping check
        if epoch >= min_epochs:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                    break

    # Save the best model's state dictionary and configuration parameters
    output_dir = os.path.join(os.path.dirname(__file__), 'Output', 'models')
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, f"best_model_{model_name}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, best_model_path)
    # mlflow.log_artifact(best_model_path, "best_model")

def save_training_curves(train_losses, train_accuracies, val_losses, val_accuracies, config, model_name, fold_num):
    '''
    Save training curves to plots and CSV.
    '''
    import pandas as pd
    
    output_dir = os.path.join(os.path.dirname(__file__), 'Output', 'training_curves')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame with training history
    curves_df = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Train_Loss': train_losses,
        'Train_Accuracy': train_accuracies,
        'Val_Loss': val_losses,
        'Val_Accuracy': val_accuracies
    })
    
    # Save to CSV
    csv_path = os.path.join(output_dir, f"training_curves_fold_{fold_num}_{model_name or 'default'}.csv")
    curves_df.to_csv(csv_path, index=False)
    
    # Create plots
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss plot
    ax1.plot(curves_df['Epoch'], curves_df['Train_Loss'], label='Training Loss', color='blue')
    ax1.plot(curves_df['Epoch'], curves_df['Val_Loss'], label='Validation Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training and Validation Loss - Fold {fold_num}')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(curves_df['Epoch'], curves_df['Train_Accuracy'], label='Training Accuracy', color='blue')
    ax2.plot(curves_df['Epoch'], curves_df['Val_Accuracy'], label='Validation Accuracy', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Training and Validation Accuracy - Fold {fold_num}')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"training_curves_fold_{fold_num}_{model_name or 'default'}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved: {plot_path}")
    return csv_path, plot_path

def test_best_model(best_result, dataset, filtered_dataset_test, model_name, output_dir):
    '''
    Tests the trained model with best hyperparameters on the test set.

    INPUTS:
        - best_result               : Result object of results of best hyperparameter configuration
        - dataset                   : Dataset of graphs
        - filtered_dataset_test     : Dataset of filtered graphs for testing
        - model_name                : Name of the model to be saved
        - output_dir                : Directory to save the output plots and results
    
    OUTPUT: 
        - acc_test          : Float of accuracy of model's predictions of test set
        - f1                : Float of F1 score of model's predictions of test set
        - roc_auc           : Float of ROC AUC score of model's predictions of test set
        - precision         : Float of precision of model's predictions of test set
        - recall            : Float of recall of model's predictions of test set
        - conf_matrix       : Confusion matrix of model's predictions of test set
        - y_true            : Numpy array of true labels of test set
        - y_prob            : Numpy array of predicted probabilities of positive class (ON) for test set
    '''

    # Initialize model with best hyperparameter for 'hidden_channels'
    best_trained_model = GNN(
        n_layers=best_result.config['n_layers'],
        dropout_rate=best_result.config['dropout_rate'],
        conv1_hidden_channels=best_result.config['conv1_hidden_channels'], 
        conv2_hidden_channels=best_result.config['conv1_hidden_channels'],
        conv3_hidden_channels=best_result.config['conv1_hidden_channels'],
        conv4_hidden_channels=best_result.config['conv1_hidden_channels'],
        dataset=dataset
    )

    # Retrieve trained model parameters from the checkpoint of the best result and load onto the initialized model
    with best_result.checkpoint.as_directory() as checkpoint_dir:
        state_dict = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pt'), weights_only=True)
        best_trained_model.load_state_dict(state_dict['model_state'])

    # Batch test dataset with best batch size
    test_loader = DataLoader(
        filtered_dataset_test, 
        best_result.config['batch_size'],
        shuffle=False
    )
    print(f"Test dataset size: {len(filtered_dataset_test)}")
    print(f"Batch size: {best_result.config['batch_size']}")
    
    # Put model in evaluation mode
    best_trained_model.eval()

    # Calculate accuracy of trained model on test set 
    correct = 0
    y_true = []
    y_pred = []
    y_prob = []
    for data in test_loader:
        out = best_trained_model(data.x, data.edge_index, data.edge_attr, data.batch)
        prob = F.softmax(out, dim=1)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())

        # Append the true and predicted labels
        y_true.append(data.y.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
        y_prob.append(prob[:, 1].detach().cpu().numpy())
    print(f"Number of y_true arrays: {len(y_true)}")
    print(f"Number of y_pred arrays: {len(y_pred)}")
    print(f"Number of y_prob arrays: {len(y_prob)}")

    # Flatten the lists into arrays
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)

    # Ensure they are of the same type
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    # y_prob = y_prob.astype(float)

    # Calculate performance metrics
    accuracy_test = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.0  # Handle case where ROC AUC cannot be computed

    # Print performance metrics
    print('Performance metrics of test set:')
    print('ROC AUC', roc_auc)
    print('Accuracy', accuracy_test)
    print('Precision', precision)
    print('Recall', recall)

    acc_test = correct/len(test_loader.dataset)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    print('Length dataset', len(test_loader.dataset))
    print('Length total', [data.y.size(0) for data in test_loader])

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    return acc_test, f1, roc_auc, precision, recall, conf_matrix, y_true, y_prob

def load_and_test_model(model_path, best_result, dataset, filtered_dataset_test):
    '''
    Loads the saved model and tests it on new data.

    INPUTS:
        - model_path            : Path to the saved model
        - best_result           : Result object of results of best hyperparameter configuration
        - dataset               : Dataset of graphs for testing
        - filtered_dataset_test : Filtered dataset for testing
    
    OUTPUT: 
        - acc_test              : Float of accuracy of model's predictions on the test data
    '''

    # Initialize the model with the saved configuration parameters
    model = GNN(
        n_layers=best_result.config['n_layers'],
        dropout_rate=best_result.config['dropout_rate'],
        conv1_hidden_channels=best_result.config['conv1_hidden_channels'], 
        conv2_hidden_channels=best_result.config['conv1_hidden_channels'],
        conv3_hidden_channels=best_result.config['conv1_hidden_channels'],
        conv4_hidden_channels=best_result.config['conv1_hidden_channels'],
        dataset=dataset
    )

    # Retrieve trained model parameters from the checkpoint of the best result and load onto the initialized model
    with best_result.checkpoint.as_directory() as checkpoint_dir:
        state_dict = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pt'), weights_only=True)
        model.load_state_dict(state_dict['model_state'])

    # Batch new data with the best batch size
    test_loader = DataLoader(
        filtered_dataset_test, 
        best_result.config['batch_size'],
        shuffle=False
    )
    
    # Put model in evaluation mode
    model.eval()

    # Calculate accuracy of trained model on new data
    correct = 0
    y_true = []
    y_pred = []
    y_prob = []
    for data in test_loader:
        out = model(data.x, data.edge_index, data.edge_attr, data.batch) #, temperature=1)
        prob = F.softmax(out, dim=1)
        pred = prob.argmax(dim=1)
        correct += int((pred == data.y).sum())

        # Append the true and predicted labels
        y_true.append(data.y.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
        y_prob.append(prob[:,1].detach().cpu().numpy())

    # Flatten the lists into arrays
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)

    # Ensure they are of the same type
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    y_prob = y_prob.astype(float)
    print('y pred', y_pred)
    print('y prob', y_prob)

    # Calculate performance metrics
    accuracy_test = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        roc_auc = 0.0  # Handle case where ROC AUC cannot be computed

    acc_test = correct / len(test_loader.dataset)
    print('Length dataset', len(test_loader.dataset))
    print('Length total', [data.y.size(0) for data in test_loader])

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Print performance metrics
    print('Performance metrics on test data:')
    print('ROC AUC', roc_auc)
    print('Accuracy', accuracy_test)
    print('F1 Score', f1)
    print('Precision', precision)
    print('Recall', recall)

    # Plot the confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['OFF', 'ON'], yticklabels=['OFF', 'ON'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # Add accuracy and F1 score in a separate text box
    plt.figtext(0.5, 0.01, f'Accuracy: {accuracy:.2f} | F1 Score: {f1:.2f}', ha='center', fontsize=12)

    plt.tight_layout()
    plt.show()

    # Plot the ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Save ROC curve values to a CSV file
    roc_data = pd.DataFrame({
        'Model': [model_path] * len(fpr),
        'False Positive Rate': fpr,
        'True Positive Rate': tpr,
        'Thresholds': thresholds
    })
    roc_output_dir = os.path.join(os.path.dirname(__file__), 'Output', 'ROC')
    os.makedirs(roc_output_dir, exist_ok=True)
    roc_data.to_csv(os.path.join(roc_output_dir, f'ROC_curve_{os.path.basename(model_path)}.csv'), index=False)

    return acc_test, f1, roc_auc, precision, recall, conf_matrix

def explain_model(best_result, filtered_dataset_complete, dataset, input_type, explain_dir):
    '''
    Loads the saved model and performs explainability based on the SubgraphX methodology on the test set.

    INPUTS:
        - best_result               : Result object of results of best hyperparameter configuration
        - filtered_dataset_complete : Complete filtered test dataset
        - dataset                   : Dataset of graphs for testing
        - input_type                : Type of input data (e.g., 'PSD', 'connectivity', etc.)
        - explain_dir               : Directory to save the explanation results
    OUTPUT: N/A
    '''

    # Select the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model with the saved configuration parameters
    model = GNN(
        n_layers=best_result.config['n_layers'],
        dropout_rate=best_result.config['dropout_rate'],
        conv1_hidden_channels=best_result.config['conv1_hidden_channels'], 
        conv2_hidden_channels=best_result.config['conv1_hidden_channels'],
        conv3_hidden_channels=best_result.config['conv1_hidden_channels'],
        conv4_hidden_channels=best_result.config['conv1_hidden_channels'],
        dataset=dataset,
        top_k=best_result.config['top_k'],
        threshold=best_result.config['threshold']
    )

    # Retrieve trained model parameters from the checkpoint of the best result and load onto the initialized model
    with best_result.checkpoint.as_directory() as checkpoint_dir:
        state_dict = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pt'), weights_only=True)
        model.load_state_dict(state_dict['model_state'])
        model.to(device)

    # Create dataloader for explainability with the filtered dataset
    explain_loader = DataLoader(
        filtered_dataset_complete,
        1,
        shuffle=False
    )

    # Generate Excel file in explainability folder
    excel_path = os.path.join(explain_dir, f"explanations_testfile_graphs.xlsx")

    edge_weights_graph = []

    for i, data in enumerate(explain_loader):
        # Move the batch to the appropriate device (CPU/GPU)
        data = data.to(device)

        # Loop through the graphs in the batch
        print(f"Graph number: {i+1}/{len(filtered_dataset_complete)}")
        graph = data
        label = graph.y.item()

        # Initialise start time for timer
        start_time = time.time()

        # Perform explainability on the graph
        avg_accuracy_normal, avg_accuracy_occluded, most_common_nodes, most_common_labels, accuracy_difference, node_frequencies, avg_fidelity_ratio, avg_fidelity_score = calculate_explain_and_accuracies(graph, model, dataset, best_result, input_type)
        print("Fidelity Score:", avg_fidelity_score)

        # Sparsity calculation
        sparsity_score = sparsity(graph, most_common_nodes)
        print("Sparsity Score:", sparsity_score)

        # Mark endtime of total run time
        end_time = time.time()
        run_time = (end_time-start_time)/60
        print(f"Runtime: {run_time:.4f} minutes.")

        # Save results to Excel file
        most_common_nodes = excel_explanation(excel_path, i, most_common_nodes, most_common_labels, node_frequencies, avg_fidelity_score, avg_fidelity_ratio, sparsity_score, avg_accuracy_normal, avg_accuracy_occluded, accuracy_difference, run_time)

        # Save a visualition of the subgraph inside the original graph
        edge_weights = extract_edge_weights(graph)
        edge_weights_graph.append(edge_weights)
        visualize_graph_with_subgraph(graph, label, most_common_nodes, dataset, input_type, explain_dir, i, edge_weights)

    # Visualize the graph with the averaged edge weights
    avg_edge_weights = average_edge_weights(edge_weights_graph)
    visualize_graph_with_subgraph(graph, 'total', most_common_nodes, dataset, input_type, explain_dir, 'all', avg_edge_weights)
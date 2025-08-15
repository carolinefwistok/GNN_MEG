import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from sklearn.model_selection import StratifiedGroupKFold
from train import train_hyperparameters, test_best_model
from model import GNN
from dataset_wrapper import LoadedGraphsDataset
from utils import edge_filtering, mst_filtering
import argparse
from mpi4py import MPI
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

import matplotlib.pyplot as plt
import numpy as np

def plot_cv_splits(splits, labels, patient_ids, model_name=None, save_path=None):
    """
    Visualize cross-validation splits, class labels, and patient groups.

    Args:
        splits: List of (train_idx, test_idx) tuples from StratifiedGroupKFold.
        labels: List of class labels for each graph.
        patient_ids: List of patient/group IDs for each graph.
        save_path: Optional path to save the plot.
    """
    n_samples = len(labels)
    n_splits = len(splits)
    cmap_data = plt.cm.Paired
    cmap_cv = plt.cm.coolwarm
    cmap_group = plt.cm.tab20

    fig, ax = plt.subplots(figsize=(12, 2 + n_splits * 0.5))
    marker_size = 100  # Thicker lines

    # CV splits
    for ii, (train_idx, test_idx) in enumerate(splits):
        # Create array with 0 for train, 1 for test, -1 for unused
        mask = np.full(n_samples, -1)
        mask[train_idx] = 0
        mask[test_idx] = 1
        ax.scatter(
            range(n_samples),
            [ii + 0.5] * n_samples,
            c=mask,
            marker='s',
            s=20,
            cmap=cmap_cv,
            label=None,
            edgecolor='none'
        )

    # Plot class labels
    classes = sorted(set(labels))
    class_handles = []
    for c in classes:
        class_handles.append(
            plt.Line2D([0], [0], color=cmap_data(c / max(classes)), marker='s', linestyle='', markersize=10, label=f'Class {c}')
        )
    ax.scatter(
        range(n_samples),
        [n_splits + 0.5] * n_samples,
        c=labels,
        marker='s',
        s=20,
        cmap=cmap_data,
        label='class',
        edgecolor='none'
    )
    # Plot patient/group IDs
    unique_groups = {g: i for i, g in enumerate(sorted(set(patient_ids)))}
    group_colors = [unique_groups[g] for g in patient_ids]
    group_handles = []
    for g, idx in unique_groups.items():
        group_handles.append(
            plt.Line2D([0], [0], color=cmap_group(idx / max(unique_groups.values())), marker='s', linestyle='', markersize=10, label=f'Patient {g}')
        )
    ax.scatter(
        range(n_samples),
        [n_splits + 1.5] * n_samples,
        c=group_colors,
        marker='s',
        s=marker_size,
        cmap=cmap_group,
        label='group',
        edgecolor='none'
    )

    ax.set(
        yticks=np.arange(n_splits + 2),
        yticklabels=[f"CV fold {i}" for i in range(n_splits)] + ["class", "group"],
        xlabel="Sample index",
        ylabel="",
        title="Cross-validation splits, class, and group"
    )

    # Legend for train/test
    train_test_handles = [
        plt.Line2D([0], [0], color=cmap_cv(0.1), marker='s', linestyle='', markersize=10, label="Training set"),
        plt.Line2D([0], [0], color=cmap_cv(0.9), marker='s', linestyle='', markersize=10, label="Testing set"),
    ]
    # Combine all handles
    handles = train_test_handles + class_handles + group_handles
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.25, 1))

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'Output', model_name, 'plots', save_path) if save_path else None
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

def cross_validate(dataset, n_splits=5):
    '''
    Splits the dataset into n_splits for cross-validation, ensuring stratification by labels and grouping by patient IDs.
    INPUTS:
        - dataset:       List of graph data objects.
        - n_splits:      Number of splits for cross-validation (default is 5).
    OUTPUTS:
        - splits:        List of train/test indices for each fold.
        - dataset:       Filtered dataset with only graphs that have labels.
        - labels:        List of labels corresponding to the graphs in the dataset.
    '''

    # Acquire labels and patient IDs from the dataset
    labels = [data.y.item() for data in dataset if data.y is not None]
    patient_ids = [data.patient_id for data in dataset if data.y is not None]
    filtered_graphs = [data for data in dataset if data.y is not None]

    # Apply StratifiedGroupKFold to ensure stratification by labels and grouping by patient IDs
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(sgkf.split(np.zeros(len(labels)), labels, groups=patient_ids))

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_labels = [labels[i] for i in train_idx]
        test_labels = [labels[i] for i in test_idx]
        
        print(f"Fold {fold_idx + 1}:")
        print(f"  Train: {len(train_labels)} samples, {sum(train_labels)} positive ({sum(train_labels)/len(train_labels)*100:.1f}%)")
        print(f"  Test: {len(test_labels)} samples, {sum(test_labels)} positive ({sum(test_labels)/len(test_labels)*100:.1f}%)")
        
        # Check for extreme values in this fold's data
        fold_train_graphs = [filtered_graphs[i] for i in train_idx]
        fold_features = torch.cat([graph.x for graph in fold_train_graphs])
        print(f"  Feature range: {fold_features.min():.3f} to {fold_features.max():.3f}")
        print(f"  Feature std: {fold_features.std():.3f}")
    
    return splits, filtered_graphs, patient_ids, labels

def plot_train_results(results, best_result, output_plot_dir, model_name=None):
    '''
    Plots the training results by plotting the losses and accuracies of both the train and validation set.
    '''
    
    # Plot accuracies for different hyperparameters (learning rates)
    plt.figure(figsize=(12, 8))
    ax = None
    
    # Get all results and plot training accuracy vs iteration for different LRs
    for result in results:
        if hasattr(result, 'metrics_dataframe') and result.metrics_dataframe is not None:
            label = f"lr={result.config['lr']:.5f}"
            if ax is None:
                ax = result.metrics_dataframe.plot("training_iteration", "train_accuracy", label=label, alpha=0.7)
            else:
                result.metrics_dataframe.plot("training_iteration", "train_accuracy", ax=ax, label=label, alpha=0.7)
    
    if ax is not None:
        ax.set_title("Train Accuracy across training iterations for all LRs")
        ax.set_xlabel("Training iteration")
        ax.set_ylabel("Training Accuracy")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        lr_plot_path = os.path.join(output_plot_dir, f"lr_comparison_{model_name or 'default'}.png")
        plt.savefig(lr_plot_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close to save memory
        print(f"Learning rate comparison plot saved: {lr_plot_path}")

    # Plot losses and accuracies for best configuration
    if hasattr(best_result, 'metrics_dataframe') and best_result.metrics_dataframe is not None:
        plt.figure(figsize=(15, 6))
        
        # Loss plot
        plt.subplot(121)
        plt.plot(best_result.metrics_dataframe['training_iteration'], 
                 best_result.metrics_dataframe['train_loss'], 
                 label='Train loss', color='blue', linewidth=2)
        plt.plot(best_result.metrics_dataframe['training_iteration'], 
                 best_result.metrics_dataframe['val_loss'], 
                 label='Validation loss', color='red', linewidth=2)
        plt.legend()
        plt.ylabel('Loss')
        plt.xlabel('Training iteration')
        plt.title('Training and Validation Loss (Best Config)')
        plt.grid(True, alpha=0.3)
        
        # Accuracy plot
        plt.subplot(122)
        plt.plot(best_result.metrics_dataframe['training_iteration'], 
                 best_result.metrics_dataframe['train_accuracy'], 
                 label='Train accuracy', color='blue', linewidth=2)
        plt.plot(best_result.metrics_dataframe['training_iteration'], 
                 best_result.metrics_dataframe['val_accuracy'], 
                 label='Validation accuracy', color='red', linewidth=2)
        plt.legend()
        plt.ylabel('Accuracy')
        plt.xlabel('Training iteration')
        plt.title('Training and Validation Accuracy (Best Config)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        best_plot_path = os.path.join(output_plot_dir, f"best_training_curves_{model_name or 'default'}.png")
        plt.savefig(best_plot_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close to save memory
        print(f"Best training curves plot saved: {best_plot_path}")

def save_results_with_metrics(results_df, best_params, best_result, output_directory, all_metrics, all_y_true, all_y_prob, model_name=None, overall_results=None):
    '''
    Saves the training results, hyperparameter configurations, and cross-validation metrics to an Excel file.
    
    INPUTS:
        - results_df: DataFrame containing the results of the training.
        - best_params: Dictionary of the best hyperparameter configuration.
        - best_result: Best result object from Ray Tune.
        - output_directory: Directory where the Excel file will be saved.
        - all_metrics: List of tuples containing (acc_test, f1, roc_auc, precision, recall) for each fold
        - model_name: Name of the model for file naming.
    
    OUTPUT: Path to the saved Excel file
    '''

    # Get ROC data
    fpr, tpr, thresholds = roc_curve(all_y_true, all_y_prob)
    roc_auc = auc(fpr, tpr)
    
    roc_df = pd.DataFrame({
        'False_Positive_Rate': fpr,
        'True_Positive_Rate': tpr,
        'Thresholds': thresholds,
        'AUC': [roc_auc] * len(fpr)
    })
    
    # Calculate confusion matrix at 0.5 threshold
    y_pred = (all_y_prob >= 0.5).astype(int)
    cm = confusion_matrix(all_y_true, y_pred)
    
    confusion_summary = pd.DataFrame({
        'Metric': ['True_Negative', 'False_Positive', 'False_Negative', 'True_Positive',
                   'Sensitivity', 'Specificity', 'Precision', 'Accuracy', 'F1_Score'],
        'Value': [
            cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1],
            cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0,
            cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0,
            cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0,
            (cm[0, 0] + cm[1, 1]) / cm.sum(),
            2 * cm[1, 1] / (2 * cm[1, 1] + cm[0, 1] + cm[1, 0]) if (2 * cm[1, 1] + cm[0, 1] + cm[1, 0]) > 0 else 0
        ]
    })
    
   # Convert all_metrics to a proper DataFrame
    metrics_names = ["Accuracy", "F1", "ROC_AUC", "Precision", "Recall"]
    cv_metrics_data = []
    
    for fold_idx, metrics in enumerate(all_metrics):
        fold_data = {'Fold': fold_idx + 1}
        for i, name in enumerate(metrics_names):
            fold_data[name] = metrics[i]
        cv_metrics_data.append(fold_data)
    
    cv_metrics_df = pd.DataFrame(cv_metrics_data)

    summary_stats_data = []
    for i, name in enumerate(metrics_names):
        metric_values = [metrics[i] for metrics in all_metrics]
        summary_stats_data.append({
            'Metric': name,
            'Mean': np.mean(metric_values),
            'Std': np.std(metric_values),
            'Min': np.min(metric_values),
            'Max': np.max(metric_values)
        })
    
    summary_stats_df = pd.DataFrame(summary_stats_data)
    best_params_df = pd.DataFrame([best_params])

    if hasattr(best_result, 'metrics_dataframe') and best_result.metrics_dataframe is not None:
        iteration_metrics_df = best_result.metrics_dataframe.copy()
    else:
        iteration_metrics_df = pd.DataFrame()

    if overall_results:
        overall_results_df = pd.DataFrame([overall_results])
    else:
        overall_results_df = pd.DataFrame()

    # Create output file path
    output_file = os.path.join(output_directory, f'training_results_{model_name}.xlsx')

    # Create results DataFrame with specified columns
    columns_to_export = [
        'config/n_layers',
        'config/dropout_rate', 
        'config/conv1_hidden_channels',
        'config/lr',
        'config/batch_size',
        'config/weight_decay',
        'config/edge_filtering',
        'config/top_k',
        'config/threshold',
        'train_accuracy',
        'val_accuracy',
        'train_loss',
        'val_loss',
    ]
    
    # Filter columns that exist in the DataFrame
    existing_columns = [col for col in columns_to_export if col in results_df.columns]
    results_df_export = results_df[existing_columns]

    # Create a Pandas Excel writer using openpyxl as the engine
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # ROC and confusion matrix data
        roc_df.to_excel(writer, sheet_name='ROC_Curve_Data', index=False)
        confusion_summary.to_excel(writer, sheet_name='Confusion_Matrix_Summary', index=False)
        
        # Overall results
        if not overall_results_df.empty:
            overall_results_df.to_excel(writer, sheet_name='Overall_Test_Results', index=False)
        
        # Write the cross-validation metrics
        cv_metrics_df.to_excel(writer, sheet_name='CV_Metrics_per_Fold', index=False)
        
        # Write the summary statistics
        summary_stats_df.to_excel(writer, sheet_name='CV_Summary_Statistics', index=False)
        
        # Write the hyperparameter tuning results
        results_df_export.to_excel(writer, sheet_name='Hyperparameter_Results', index=False)
        
        # Write the best parameters
        best_params_df.to_excel(writer, sheet_name='Best_Parameters', index=False)
        
        # Write the training curves for best model (if available)
        if not iteration_metrics_df.empty:
            iteration_metrics_df.to_excel(writer, sheet_name='Best_Training_Curves', index=False)
    
    print(f'Combined results saved to {output_file}')
    sheet_names = ['Overall_Test_Results', 'CV_Metrics_per_Fold', 'CV_Summary_Statistics', 
                   'Hyperparameter_Results', 'Best_Parameters', 'Best_Training_Curves']
    print(f'Excel file contains {len(sheet_names)} sheets: {", ".join(sheet_names)}')
    
    return output_file

def save_roc_data_for_plotting(all_y_true, all_y_prob, output_dir, model_name):
    """
    Save only the essential ROC curve data needed for plotting.
    
    INPUTS:
        - all_y_true: Array of true labels from all folds
        - all_y_prob: Array of predicted probabilities from all folds
        - output_dir: Directory to save the Excel file
        - model_name: Name of the model for file naming
    
    OUTPUT: Path to the saved Excel file
    """
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(all_y_true, all_y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Create minimal ROC curve DataFrame for plotting
    roc_data = pd.DataFrame({
        'FPR': fpr,
        'TPR': tpr,
        'Thresholds': thresholds
    })
    
    # Add summary information as a separate sheet
    summary_data = pd.DataFrame({
        'Metric': ['AUC', 'Total_Samples', 'Positive_Samples', 'Negative_Samples'],
        'Value': [
            roc_auc,
            len(all_y_true),
            np.sum(all_y_true),
            len(all_y_true) - np.sum(all_y_true)
        ]
    })
    
    # Save to Excel with minimal sheets
    output_file = os.path.join(output_dir, f'roc_data_{model_name}.xlsx')
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Essential ROC data for plotting
        roc_data.to_excel(writer, sheet_name='ROC_Data', index=False)
        
        # Summary metrics
        summary_data.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f'ROC plotting data saved to {output_file}')
    return output_file

def plot_fold_results_combined(all_results, all_best_results, output_plot_dir, model_name=None):
    '''
    Plot training curves for all folds showing the best result from each fold.
    
    INPUTS:
        - all_results: List of ResultGrid objects from each fold
        - all_best_results: List of best result objects from each fold
        - output_plot_dir: Directory to save plots
        - model_name: Name of the model for saving files
    '''
    
    plt.figure(figsize=(15, 10))
    
    # Training accuracy curves for best config from each fold
    plt.subplot(2, 2, 1)
    for fold_idx, best_result in enumerate(all_best_results):
        if hasattr(best_result, 'metrics_dataframe') and best_result.metrics_dataframe is not None:
            plt.plot(best_result.metrics_dataframe['training_iteration'], 
                    best_result.metrics_dataframe['train_accuracy'], 
                    label=f'Fold {fold_idx+1}', alpha=0.7, linewidth=2)
    plt.xlabel('Training Iteration')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy - Best Config per Fold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Validation accuracy curves for best config from each fold
    plt.subplot(2, 2, 2)
    for fold_idx, best_result in enumerate(all_best_results):
        if hasattr(best_result, 'metrics_dataframe') and best_result.metrics_dataframe is not None:
            plt.plot(best_result.metrics_dataframe['training_iteration'], 
                    best_result.metrics_dataframe['val_accuracy'], 
                    label=f'Fold {fold_idx+1}', alpha=0.7, linewidth=2)
    plt.xlabel('Training Iteration')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy - Best Config per Fold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Training loss curves for best config from each fold
    plt.subplot(2, 2, 3)
    for fold_idx, best_result in enumerate(all_best_results):
        if hasattr(best_result, 'metrics_dataframe') and best_result.metrics_dataframe is not None:
            plt.plot(best_result.metrics_dataframe['training_iteration'], 
                    best_result.metrics_dataframe['train_loss'], 
                    label=f'Fold {fold_idx+1}', alpha=0.7, linewidth=2)
    plt.xlabel('Training Iteration')
    plt.ylabel('Training Loss')
    plt.title('Training Loss - Best Config per Fold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Validation loss curves for best config from each fold
    plt.subplot(2, 2, 4)
    for fold_idx, best_result in enumerate(all_best_results):
        if hasattr(best_result, 'metrics_dataframe') and best_result.metrics_dataframe is not None:
            plt.plot(best_result.metrics_dataframe['training_iteration'], 
                    best_result.metrics_dataframe['val_loss'], 
                    label=f'Fold {fold_idx+1}', alpha=0.7, linewidth=2)
    plt.xlabel('Training Iteration')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss - Best Config per Fold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    fold_plot_path = os.path.join(output_plot_dir, f"all_folds_training_curves_{model_name or 'default'}.png")
    plt.savefig(fold_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"All folds training curves plot saved: {fold_plot_path}")

def main(processed_analysis_dir, model_name=None, input_type='scout'):
    '''
    Main function to load the dataset, split it into train/test sets, train the GNN model,
    tune hyperparameters, and evaluate the model on the test set.

    INPUTS:
        - processed_analysis_dir:   Path to the processed_<analysis> directory containing .pt graph files.
        - model_name:               Optional name for saving the trained model/results.
        - input_type:               Type of input data, either 'scout' or 'fif'.
    OUTPUT: N/A
    '''

    print('Initializing MPI and loading dataset...')
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Load dataset from saved processed directory
    print(f"Rank {rank}: Loading graphs from {processed_analysis_dir}")
    dataset = LoadedGraphsDataset(processed_analysis_dir, add_patient_ids=False)
    print(f"Rank {rank}: Loaded {len(dataset)} graphs.")

    print("Patient distribution:", dataset.get_patient_distribution())
    print("File distribution:", dataset.get_file_distribution())
    print("Stimulation distribution:", dataset.get_stimulation_distribution())

    dataset.verify_patient_grouping()

    print(f"Number of features for dataset: {dataset.num_node_features}")
    print(f"Number of classes for dataset: {dataset.num_classes}")

    if len(dataset) == 0:
        print(f"Rank {rank}: No valid graphs found in {processed_analysis_dir}.")
        return

    # Specify output directories
    output_dir = os.path.join(os.path.dirname(__file__), 'Output', model_name or "gnn_model")
    os.makedirs(output_dir, exist_ok=True)
    output_plot_dir = os.path.join(os.path.dirname(__file__), 'Output', model_name or "gnn_model", 'plots')
    os.makedirs(output_plot_dir, exist_ok=True)

    # Apply cross-validation to the dataset to acquire train/test splits
    splits, filtered_graphs, patient_ids, labels = cross_validate(dataset, n_splits=5)

    # Plot cross-validation splits
    plot_cv_splits(splits, labels, patient_ids, model_name=model_name, save_path="cv_splits.png")
    test_set_info = {
        'total_original_graphs': len(dataset),
        'total_valid_graphs': len(filtered_graphs),
        'total_patients': len(set(patient_ids)),
        'class_distribution': {
            'total_on': sum(labels),
            'total_off': len(labels) - sum(labels),
            'percentage_on': sum(labels) / len(labels) * 100
        },
        'fold_details': []
    }

    # Print detailed test set information
    print(f"\n=== DATASET AND TEST SET ANALYSIS ===")
    print(f"Original dataset size: {test_set_info['total_original_graphs']}")
    print(f"Valid graphs (with labels): {test_set_info['total_valid_graphs']}")
    print(f"Graphs filtered out: {test_set_info['total_original_graphs'] - test_set_info['total_valid_graphs']}")
    print(f"Total patients: {test_set_info['total_patients']}")
    print(f"Overall class distribution:")
    print(f"  ON: {test_set_info['class_distribution']['total_on']} ({test_set_info['class_distribution']['percentage_on']:.1f}%)")
    print(f"  OFF: {test_set_info['class_distribution']['total_off']} ({100 - test_set_info['class_distribution']['percentage_on']:.1f}%)")
    
    print(f"\nPer-fold test set details:")
    total_test_samples = 0
    for fold_info in test_set_info['fold_details']:
        print(f"  Fold {fold_info['fold']}: {fold_info['test_size']} samples from {fold_info['test_patients']} patients")
        print(f"    ON: {fold_info['test_on']} ({fold_info['test_percentage_on']:.1f}%), OFF: {fold_info['test_off']}")
        total_test_samples += fold_info['test_size']
    
    print(f"Total test samples across all folds: {total_test_samples}")
    print(f"Average test set size per fold: {total_test_samples / len(splits):.1f}")
    
    # Save test set information
    test_info_df = pd.DataFrame(test_set_info['fold_details'])
    test_info_df.to_csv(os.path.join(output_dir, f"test_set_info_{model_name}.csv"), index=False)
    
    # Also save summary
    summary_info = {
        'metric': ['Original_Graphs', 'Valid_Graphs', 'Filtered_Out', 'Total_Patients', 
                  'Total_ON', 'Total_OFF', 'Total_Test_Samples', 'Avg_Test_Per_Fold'],
        'value': [
            test_set_info['total_original_graphs'],
            test_set_info['total_valid_graphs'],
            test_set_info['total_original_graphs'] - test_set_info['total_valid_graphs'],
            test_set_info['total_patients'],
            test_set_info['class_distribution']['total_on'],
            test_set_info['class_distribution']['total_off'],
            total_test_samples,
            total_test_samples / len(splits)
        ]
    }
    summary_df = pd.DataFrame(summary_info)
    summary_df.to_csv(os.path.join(output_dir, f"dataset_summary_{model_name}.csv"), index=False)
    print(f"=== END DATASET ANALYSIS ===\n")

    all_metrics = []
    all_y_true = []
    all_y_prob = []
    all_results = []
    all_best_results = []

    # Assign folds to ranks for parallel processing
    folds_for_this_rank = [i for i in range(len(splits)) if i % size == rank]
    print(f"Rank {rank} will process folds: {folds_for_this_rank}")

    for fold in folds_for_this_rank:
        print(f"\n=== Fold {fold+1} ===")
        # Get train and test indices for the current fold
        train_idx, test_idx = splits[fold]

        # For each fold, create train and test datasets
        dataset_train = [filtered_graphs[i] for i in train_idx]
        dataset_test = [filtered_graphs[i] for i in test_idx]
        y_train = [labels[i] for i in train_idx]
        y_test = [labels[i] for i in test_idx]

        print(f"Fold {fold+1} - Initial feature check:")
        valid_train_count = sum(1 for graph in dataset_train if graph.x is not None and not torch.isnan(graph.x).any())
        print(f"  Valid training graphs: {valid_train_count}/{len(dataset_train)}")
        
        if valid_train_count > 0:
            train_features = torch.cat([graph.x for graph in dataset_train if graph.x is not None and not torch.isnan(graph.x).any()])
            print(f"  Feature range: {train_features.min():.3f} to {train_features.max():.3f}")
            print(f"  Feature std: {train_features.std():.3f}")
        else:
            print(f"  No valid training graphs found!")

        if len(dataset_train) == 0:
            print(f"Skipping Fold {fold+1} - no valid training data")
            continue

        # Train the model with hyperparameter tuning
        results, best_result, best_params = train_hyperparameters(dataset, dataset_train, y_train, model_name=model_name)

        # Store results for later plotting
        all_results.append(results)
        all_best_results.append(best_result)

        # Plot training results for this fold
        plot_train_results(results, best_result, output_plot_dir, f"{model_name}_fold_{fold+1}")
        
        # Save results for this fold
        results_df = results.get_dataframe()

        # Apply edge filtering based on the best hyperparameters found
        if best_result.config['edge_filtering'] == 'Threshold_TopK':
            filtered_dataset_train = [edge_filtering(graph, best_result.config['top_k'], best_result.config['threshold']) for graph in dataset_train]
            filtered_dataset_test = [edge_filtering(graph, best_result.config['top_k'], best_result.config['threshold']) for graph in dataset_test]
        elif best_result.config['edge_filtering'] == 'MST':
            filtered_dataset_train = [mst_filtering(graph) for graph in dataset_train]
            filtered_dataset_test = [mst_filtering(graph) for graph in dataset_test]
        else:
            filtered_dataset_train = dataset_train
            filtered_dataset_test = dataset_test

        # Test the best model on the test set and save output
        acc_test, f1, roc_auc, precision, recall, conf_matrix, y_true, y_prob = test_best_model(
            best_result, dataset, filtered_dataset_test, model_name or "gnn_model", output_dir
        )
        all_metrics.append((acc_test, f1, roc_auc, precision, recall))
        all_y_true.extend(y_true)
        all_y_prob.extend(y_prob)

        print(f"Rank {rank} Fold {fold+1} results:")
        print(f"-- Accuracy: {acc_test:.3f}")
        print(f"-- F1 Score: {f1:.3f}")
        print(f"-- ROC AUC: {roc_auc:.3f}")
        print(f"-- Precision: {precision:.3f}")
        print(f"-- Recall: {recall:.3f}")
        print(f"-- Confusion Matrix:\n{conf_matrix}")

    # Gather results from all ranks to rank 0
    all_metrics_gathered = comm.gather(all_metrics, root=0)
    all_y_true_gathered = comm.gather(all_y_true, root=0)
    all_y_prob_gathered = comm.gather(all_y_prob, root=0)
    all_results_gathered = comm.gather(all_results, root=0)
    all_best_results_gathered = comm.gather(all_best_results, root=0)
    
    if rank == 0:
        # Flatten lists
        all_metrics = [item for sublist in all_metrics_gathered for item in sublist]
        all_y_true = [item for sublist in all_y_true_gathered for item in sublist]
        all_y_prob = [item for sublist in all_y_prob_gathered for item in sublist]
        all_results = [item for sublist in all_results_gathered for item in sublist]
        all_best_results = [item for sublist in all_best_results_gathered for item in sublist]

        # Plot combined results from all folds
        plot_fold_results_combined(all_results, all_best_results, output_plot_dir, model_name)
       
        # Print average results over all folds
        all_metrics = np.array(all_metrics)
        print("\n=== Cross-validated Results (mean ± std) ===")
        metrics_names = ["Accuracy", "F1", "ROC AUC", "Precision", "Recall"]
        overall_results = {}
        for i, name in enumerate(metrics_names):
            # print(f"{name}: {all_metrics[:,i].mean():.3f} ± {all_metrics[:,i].std():.3f}")
            mean_val = all_metrics[:,i].mean()
            std_val = all_metrics[:,i].std()
            overall_results[f"{name}_mean"] = mean_val
            overall_results[f"{name}_std"] = std_val
            print(f"{name}: {mean_val:.3f} ± {std_val:.3f}")

        overall_results['n_folds'] = len(all_metrics)
        overall_results['total_samples'] = len(all_y_true)
        overall_results['model_name'] = model_name
        overall_results['input_type'] = input_type
        overall_results['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        overall_results_df = pd.DataFrame([overall_results])
        overall_results_df.to_csv(os.path.join(output_dir, f"overall_results_{model_name}.csv"), index=False)

        # Convert to numpy arrays
        all_y_true = np.array(all_y_true)
        all_y_prob = [item for sublist in all_y_prob for item in (sublist if isinstance(sublist, (list, np.ndarray)) else [sublist])]
        all_y_prob = np.array(all_y_prob)

        # Save ROC curve and confusion matrix to Excel
        roc_excel_file = save_roc_data_for_plotting(
            all_y_true, 
            all_y_prob, 
            output_dir, 
            model_name or "gnn_model"
        )

        # ROC curve (assuming binary classification)
        fpr, tpr, thresholds = roc_curve(all_y_true, all_y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (All Folds)')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_plot_dir, "roc_curve_all_folds.png"))
        plt.close()

        # Confusion matrix (using 0.5 threshold)
        y_pred = (all_y_prob >= 0.5).astype(int)
        cm = confusion_matrix(all_y_true, y_pred)
        np.save(os.path.join(output_plot_dir, "confusion_matrix_all_folds.npy"), cm)

        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix (All Folds)')
        plt.colorbar()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), 
                        ha="center", va="center", 
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=14, fontweight='bold')
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['OFF', 'ON'])
        plt.yticks(tick_marks, ['OFF', 'ON'])
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.savefig(os.path.join(output_plot_dir, "confusion_matrix_all_folds.png"))
        plt.close()

        # Save combined results
        if all_results:
            combined_results_df = pd.concat([results.get_dataframe() for results in all_results], ignore_index=True)
            combined_results_df['fold'] = [i//len(all_results[0]) for i in range(len(combined_results_df))]
            
            # Save everything to one comprehensive Excel file
            combined_output_file = save_results_with_metrics(
                combined_results_df, 
                all_best_results[0].config, 
                all_best_results[0], 
                output_dir, 
                all_metrics,
                all_y_true,
                all_y_prob,
                f"{model_name}_combined",
                overall_results
            )
        print(f"ROC curve saved to {os.path.join(output_plot_dir, 'roc_curve_all_folds.png')}")
        print(f"Confusion matrix saved to {os.path.join(output_plot_dir, 'confusion_matrix_all_folds.png')}")
        print(f"Training curves saved for all folds")

        # Find the best result across all folds
        best_overall_result = None
        best_overall_val_accuracy = 0
        best_fold_idx = 0
        
        for fold_idx, best_result in enumerate(all_best_results):
            val_accuracy = best_result.metrics.get('val_accuracy', 0)
            if val_accuracy > best_overall_val_accuracy:
                best_overall_val_accuracy = val_accuracy
                best_overall_result = best_result
                best_fold_idx = fold_idx
        
        if best_overall_result:
            print(f"\n=== SAVING BEST MODEL ===")
            print(f"Best model from Fold {best_fold_idx + 1} with val_accuracy: {best_overall_val_accuracy:.4f}")
            
            # Save the best model weights
            best_model_path = os.path.join(output_dir, f"best_model{model_name}.pt")
            with best_overall_result.checkpoint.as_directory() as checkpoint_dir:
                checkpoint_data = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pt'), weights_only=True)
                
                # Save model state dict and config for easy loading
                torch.save({
                    'model_state_dict': checkpoint_data['model_state'],
                    'config': best_overall_result.config,
                    'val_accuracy': best_overall_val_accuracy,
                    'fold': best_fold_idx + 1,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }, best_model_path)
            
            # Save the best result object for compatibility with test_saved_model.py
            best_result_path = os.path.join(output_dir, f"best_result{model_name}.pt")
            torch.save({
                'best_result': best_overall_result,
                'best_params': best_overall_result.config,
                'val_accuracy': best_overall_val_accuracy,
                'fold': best_fold_idx + 1
            }, best_result_path)
            
            print(f"Best model saved to: {best_model_path}")
            print(f"Best result saved to: {best_result_path}")
            print(f"Model config: {best_overall_result.config}")

if __name__ == "__main__":
    ''' Main entry point for the script. Parses command line arguments and calls the main function. '''
    print("Starting GNN script...")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train and test GNN on processed graph dataset.")
    parser.add_argument("--processed_dir", type=str, required=True, help="Path to processed_<analysis> folder with .pt graph files.")
    parser.add_argument("--model_name", type=str, default=None, help="Name for saving the trained model/results.")
    parser.add_argument("--input_type", type=str, default="scout", choices=["scout", "fif"], help="Input type (scout or fif).")
    args = parser.parse_args()

    # Call the main function with parsed arguments
    print(f"Running GNN with processed_dir={args.processed_dir}, model_name={args.model_name}, input_type={args.input_type}")
    main(args.processed_dir, model_name=args.model_name, input_type=args.input_type)
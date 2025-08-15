import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import argparse
from datetime import datetime
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

from model import GNN
from dataset_wrapper import LoadedGraphsDataset
from utils import edge_filtering, mst_filtering


def load_best_result(best_result_path):
    """
    Load the saved best result and hyperparameters.
    
    INPUTS:
        - best_result_path: Path to the saved best result (.pt file)
    
    OUTPUT:
        - best_result: Best result object
        - best_params: Best hyperparameter configuration
    """
    print(f"Loading best result from: {best_result_path}")
    
    checkpoint = torch.load(best_result_path, map_location='cpu', weights_only=False)
    
    if isinstance(checkpoint, dict):
        best_result = checkpoint.get('best_result')
        best_params = checkpoint.get('best_params', checkpoint.get('config'))
    else:
        best_result = checkpoint
        best_params = getattr(checkpoint, 'config', None)
    
    print('Best result:', best_result)
    print('Best params:', best_params)
    
    return best_result, best_params


def filter_entire_dataset(dataset, best_result):
    """
    Filter the entire dataset using the best hyperparameters for edge filtering.
    
    INPUTS:
        - dataset: Dataset or list of graphs
        - best_result: Best result object containing hyperparameters
    
    OUTPUT:
        - filtered_dataset: Filtered dataset
        - num_edges: Total number of edges in filtered dataset
    """
    # Convert dataset to list if it's a LoadedGraphsDataset
    if hasattr(dataset, 'graphs'):
        graphs_list = [graph for graph in dataset if graph.y is not None]
    else:
        graphs_list = [graph for graph in dataset if graph.y is not None]
    
    # Get filtering parameters from best result
    config = best_result.config if hasattr(best_result, 'config') else best_result
    
    # Apply edge filtering
    if config.get('edge_filtering') == 'Threshold_TopK':
        print(f"Applying Threshold_TopK filtering with top_k={config.get('top_k')} and threshold={config.get('threshold')}")
        filtered_dataset = [edge_filtering(graph, config.get('top_k'), config.get('threshold')) for graph in graphs_list]
    elif config.get('edge_filtering') == 'MST':
        print("Applying MST filtering")
        filtered_dataset = [mst_filtering(graph) for graph in graphs_list]
    else:
        print("No edge filtering applied")
        filtered_dataset = graphs_list
    
    # Count total edges
    num_edges = sum(graph.num_edges for graph in filtered_dataset)
    
    return filtered_dataset, num_edges


def load_and_test_model(model_path, best_result, dataset, filtered_dataset_test):
    """
    Loads the saved model and tests it on new data.

    INPUTS:
        - model_path: Path to the saved model
        - best_result: Result object of results of best hyperparameter configuration
        - dataset: Dataset of graphs for testing
        - filtered_dataset_test: Filtered dataset for testing
    
    OUTPUT: 
        - acc_test: Float of accuracy of model's predictions on the test data
        - roc_auc: ROC AUC score
        - precision: Precision score
        - recall: Recall score
        - conf_matrix: Confusion matrix
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the model with the saved configuration parameters
    config = best_result.config if hasattr(best_result, 'config') else best_result
    
    model = GNN(
        n_layers=config['n_layers'],
        dropout_rate=config['dropout_rate'],
        conv1_hidden_channels=config['conv1_hidden_channels'], 
        conv2_hidden_channels=config['conv1_hidden_channels'],
        conv3_hidden_channels=config['conv1_hidden_channels'],
        conv4_hidden_channels=config['conv1_hidden_channels'],
        dataset=dataset
    )
    
    # Retrieve trained model parameters from the checkpoint of the best result and load onto the initialized model
    if hasattr(best_result, 'checkpoint'):
        # Ray Tune result object
        with best_result.checkpoint.as_directory() as checkpoint_dir:
            state_dict = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pt'), weights_only=True)
            model.load_state_dict(state_dict['model_state'])
    else:
        # Direct model loading
        print(f"Loading model weights from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()

    # Batch new data with the best batch size
    test_loader = DataLoader(
        filtered_dataset_test, 
        config['batch_size'],
        shuffle=False
    )
    
    # Calculate accuracy of trained model on new data
    correct = 0
    total_samples = 0
    y_true = []
    y_pred = []
    y_prob = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch, temperature=1)
            prob = F.softmax(out, dim=1)
            pred = prob.argmax(dim=1)

            # Count correct predictions and total samples
            correct += int((pred == data.y).sum())
            total_samples += data.y.size(0)

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
    accuracy_test_sklearn = accuracy_score(y_true, y_pred)
    accuracy_test_manual = correct / total_samples
    print(f'Accuracy (sklearn): {accuracy_test_sklearn}')
    print(f'Accuracy (manual): {accuracy_test_manual}')

    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.0  # Handle case where ROC AUC cannot be computed

    acc_test = accuracy_test_sklearn

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Print performance metrics
    print('Performance metrics on test data:')
    print('Accuracy', acc_test)
    print('F1 Score', f1)
    print('ROC AUC', roc_auc)
    print('Precision', precision)
    print('Recall', recall)

    # CRITICAL: Add probability distribution analysis
    print(f'\n=== PROBABILITY DISTRIBUTION ANALYSIS ===')
    print(f'y_prob shape: {y_prob.shape}')
    print(f'y_prob range: [{y_prob.min():.4f}, {y_prob.max():.4f}]')
    print(f'y_prob mean: {y_prob.mean():.4f}')
    print(f'y_prob median: {np.median(y_prob):.4f}')
    print(f'y_prob std: {y_prob.std():.4f}')
    
    # Check class-specific probability distributions
    prob_class_0 = y_prob[y_true == 0]  # Probabilities for actual OFF samples
    prob_class_1 = y_prob[y_true == 1]  # Probabilities for actual ON samples
    
    print(f'\nCLASS-SPECIFIC PROBABILITIES:')
    print(f'Class OFF (0) - Count: {len(prob_class_0)}, Mean prob: {prob_class_0.mean():.4f}, Std: {prob_class_0.std():.4f}')
    print(f'Class ON (1)  - Count: {len(prob_class_1)}, Mean prob: {prob_class_1.mean():.4f}, Std: {prob_class_1.std():.4f}')
    
    # Check for extreme values
    very_low_prob = np.sum(y_prob < 0.1)
    very_high_prob = np.sum(y_prob > 0.9)
    print(f'Samples with prob < 0.1: {very_low_prob} ({very_low_prob/len(y_prob)*100:.1f}%)')
    print(f'Samples with prob > 0.9: {very_high_prob} ({very_high_prob/len(y_prob)*100:.1f}%)')
    
    # Histogram of probabilities
    print(f'\nPROBABILITY HISTOGRAM:')
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(y_prob, bins=bins)
    for i, (bin_start, bin_end, count) in enumerate(zip(bins[:-1], bins[1:], hist)):
        print(f'  {bin_start:.1f}-{bin_end:.1f}: {count:3d} samples ({count/len(y_prob)*100:5.1f}%)')
    
    # Check if model is actually learning
    print(f'\nMODEL DISCRIMINATION CHECK:')
    if len(prob_class_0) > 0 and len(prob_class_1) > 0:
        separation = prob_class_1.mean() - prob_class_0.mean()
        print(f'Mean probability difference (ON - OFF): {separation:.4f}')
        if separation < 0.1:
            print('⚠️  WARNING: Poor class separation! Model may not be learning properly.')
        elif separation < 0:
            print('🚨 CRITICAL: Negative separation! Model is predicting backwards!')
    
    # Original threshold optimization code...
    thresholds = np.arange(0.05, 0.95, 0.05)  # Extended range for better analysis
    f1_scores = []
    acc_scores = []
    
    print(f'\n=== DETAILED THRESHOLD ANALYSIS ===')
    print(f"{'Threshold':<10} {'Accuracy':<10} {'F1':<10} {'Precision':<10} {'Recall':<10} {'TP':<5} {'FP':<5} {'TN':<5} {'FN':<5}")
    print("-" * 75)
    
    for thresh in thresholds:
        y_pred_thresh = (y_prob >= thresh).astype(int)
        
        # Calculate confusion matrix components
        tp = np.sum((y_true == 1) & (y_pred_thresh == 1))
        fp = np.sum((y_true == 0) & (y_pred_thresh == 1))
        tn = np.sum((y_true == 0) & (y_pred_thresh == 0))
        fn = np.sum((y_true == 1) & (y_pred_thresh == 0))
        
        acc_thresh = accuracy_score(y_true, y_pred_thresh)
        f1_thresh = f1_score(y_true, y_pred_thresh, average='weighted')
        prec_thresh = precision_score(y_true, y_pred_thresh, average='weighted', zero_division=0)
        rec_thresh = recall_score(y_true, y_pred_thresh, average='weighted')
        
        f1_scores.append(f1_thresh)
        acc_scores.append(acc_thresh)
        
        print(f"{thresh:<10.2f} {acc_thresh:<10.3f} {f1_thresh:<10.3f} {prec_thresh:<10.3f} {rec_thresh:<10.3f} {tp:<5d} {fp:<5d} {tn:<5d} {fn:<5d}")
    
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]
    best_f1 = f1_scores[best_f1_idx]
    
    print(f'\nBest threshold: {best_threshold:.2f} with F1: {best_f1:.3f}')
    
    # Validate the suspicious low threshold
    if best_threshold < 0.2:
        print(f'\n🚨 SUSPICIOUS LOW THRESHOLD DETECTED! 🚨')
        print(f'This suggests one of the following problems:')
        print(f'1. Model is severely biased toward class 0')
        print(f'2. Data leakage or mislabeling')
        print(f'3. Model architecture/training issue')
        print(f'4. Probability calibration problem')
        
        # Additional diagnostics
        pred_counts_default = np.bincount(y_pred)
        pred_counts_optimal = np.bincount((y_prob >= best_threshold).astype(int))
        
        print(f'\nPREDICTION COUNTS:')
        print(f'Default (0.5):  OFF={pred_counts_default[0] if len(pred_counts_default) > 0 else 0}, ON={pred_counts_default[1] if len(pred_counts_default) > 1 else 0}')
        print(f'Optimal ({best_threshold:.2f}): OFF={pred_counts_optimal[0] if len(pred_counts_optimal) > 0 else 0}, ON={pred_counts_optimal[1] if len(pred_counts_optimal) > 1 else 0}')
        print(f'Actual:         OFF={len(y_true) - sum(y_true)}, ON={sum(y_true)}')
    
    # Rest of your existing code...
    y_pred_optimal = (y_prob >= best_threshold).astype(int)
    
    acc_default = accuracy_score(y_true, y_pred)
    acc_optimal = accuracy_score(y_true, y_pred_optimal)
    f1_default = f1_score(y_true, y_pred, average='weighted')
    f1_optimal = f1_score(y_true, y_pred_optimal, average='weighted')
    
    print(f"\n=== FINAL THRESHOLD COMPARISON ===")
    print(f"Default (0.5):  Acc={acc_default:.3f}, F1={f1_default:.3f}")
    print(f"Optimal ({best_threshold:.2f}): Acc={acc_optimal:.3f}, F1={f1_optimal:.3f}")
    print(f"Improvement:    Acc={acc_optimal-acc_default:+.3f}, F1={f1_optimal-f1_default:+.3f}")
    

    return acc_test, roc_auc, precision, recall, conf_matrix, f1, y_true, y_pred

def save_metrics_to_excel(metrics, dataset_info, best_params, output_directory, model_name):
    """
    Save test metrics to an Excel file with detailed information.
    
    INPUTS:
        - metrics: Dictionary of performance metrics
        - dataset_info: Dictionary with dataset information
        - best_params: Best hyperparameters used for testing
        - output_directory: Directory to save the Excel file
        - model_name: Name of the model for filename
    """
    os.makedirs(output_directory, exist_ok=True)
    
    # Create a DataFrame for results
    results_data = {
        'Metric': ['Test Accuracy', 'F1 Score', 'ROC AUC', 'Precision', 'Recall', 'Number of Test Samples'],
        'Value': [
            metrics['Test Accuracy'],
            metrics['F1 Score'], 
            metrics['ROC AUC'],
            metrics['Precision'],
            metrics['Recall'],
            dataset_info['num_test_samples']
        ],
        'Description': [
            'Overall accuracy on test set',
            'Weighted F1 score',
            'Area under ROC curve',
            'Macro-averaged precision',
            'Weighted recall',
            'Total number of test samples'
        ]
    }
    
    results_df = pd.DataFrame(results_data)
    
    # Create a DataFrame for dataset information
    dataset_data = {
        'Dataset Information': [
            'Total Graphs Loaded',
            'Valid Test Graphs',
            'Filtered Out Graphs',
            'Class Distribution (ON)',
            'Class Distribution (OFF)',
            'Test Dataset Directory',
            'Model Used',
            'Test Date'
        ],
        'Value': [
            dataset_info['total_loaded'],
            dataset_info['num_test_samples'],
            dataset_info['total_loaded'] - dataset_info['num_test_samples'],
            dataset_info['class_on'],
            dataset_info['class_off'],
            dataset_info['test_dir'],
            model_name,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ]
    }
    
    dataset_df = pd.DataFrame(dataset_data)
    
    # Create a DataFrame for hyperparameters
    hyperparams_data = {
        'Hyperparameter': list(best_params.keys()),
        'Value': list(best_params.values())
    }
    
    hyperparams_df = pd.DataFrame(hyperparams_data)
    
    # Create a DataFrame for confusion matrix
    conf_matrix = metrics['Confusion Matrix']
    confusion_df = pd.DataFrame(
        conf_matrix,
        columns=['Predicted OFF', 'Predicted ON'],
        index=['Actual OFF', 'Actual ON']
    )
    
    # Add row and column totals
    confusion_df['Total'] = confusion_df.sum(axis=1)
    confusion_df.loc['Total'] = confusion_df.sum()
    
    # Save to Excel
    excel_filename = f"test_results_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    excel_path = os.path.join(output_directory, excel_filename)
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Write each sheet
        results_df.to_excel(writer, sheet_name='Test_Results', index=False)
        dataset_df.to_excel(writer, sheet_name='Dataset_Info', index=False)
        hyperparams_df.to_excel(writer, sheet_name='Hyperparameters', index=False)
        confusion_df.to_excel(writer, sheet_name='Confusion_Matrix', index=True)
        workbook = writer.book
        
        # Format Test Results sheet
        worksheet = writer.sheets['Test_Results']
        worksheet.column_dimensions['A'].width = 20
        worksheet.column_dimensions['B'].width = 15
        worksheet.column_dimensions['C'].width = 30
        
        # Format Dataset Info sheet
        worksheet = writer.sheets['Dataset_Info']
        worksheet.column_dimensions['A'].width = 25
        worksheet.column_dimensions['B'].width = 30
        
        # Format Hyperparameters sheet
        worksheet = writer.sheets['Hyperparameters']
        worksheet.column_dimensions['A'].width = 25
        worksheet.column_dimensions['B'].width = 20
    
    print(f"Excel results saved to: {excel_path}")
    return excel_path

def main():
    """Main function following your existing pattern."""
    parser = argparse.ArgumentParser(description="Test a saved GNN model on new dataset")
    parser.add_argument("--model_name", type=str, required=True, 
                       help="Model name (will be used to find best_model{model_name}.pt and best_result{model_name}.pt)")
    parser.add_argument("--test_dataset_dir", type=str, required=True,
                       help="Path to the test dataset directory (processed graphs)")
    parser.add_argument("--output_dir", type=str, default="./Output",
                       help="Base output directory")
    
    args = parser.parse_args()
    
    # Set up paths following your pattern
    model_path = os.path.join(os.path.dirname(__file__), 'Output', args.model_name, f'best_model{args.model_name}.pt')
    best_result_path = os.path.join(os.path.dirname(__file__), 'Output', args.model_name, f'best_result{args.model_name}.pt')

    print('Loading saved model...')
    # Load the saved best result and hyperparameters
    best_result, best_params = load_best_result(best_result_path)
    
    # Load test dataset
    print(f"Loading test dataset from: {args.test_dataset_dir}")
    dataset = LoadedGraphsDataset(args.test_dataset_dir, add_patient_ids=False)
    print(f"Loaded {len(dataset)} graphs for testing")
    
    if len(dataset) == 0:
        print("No valid graphs found in test dataset!")
        return
    
    # When applicable, split the dataset into a train and test set
    dataset_test = dataset
    print(f"Number of test graphs: {len(dataset_test)}")
    
    # Filter the entire dataset using the best hyperparameters for edge filtering
    filtered_dataset_complete, num_edges = filter_entire_dataset(dataset_test, best_result)
    print(f'Number of edges in filtered dataset: {num_edges}')
    print(f'Number of graphs in filtered dataset: {len(filtered_dataset_complete)}')
    
    # Load and test a saved model on the current dataset
    acc_test, roc_auc, precision, recall, conf_matrix, f1, y_true, y_pred = load_and_test_model(
        model_path, best_result, dataset, filtered_dataset_complete
    )
    print(f'Test accuracy: {acc_test}')
    
    # Prepare dataset information for saving
    class_on = int(sum(y_true))
    class_off = int(len(y_true) - class_on)
    dataset_info = {
        "total_loaded": len(dataset),
        "num_test_samples": len(y_true),
        "class_on": class_on,
        "class_off": class_off,
        "test_dir": args.test_dataset_dir
    }

    # Save metrics to text file
    metrics = {
        "Test Accuracy": acc_test,
        "F1 Score": f1,
        "ROC AUC": roc_auc,
        "Precision": precision,
        "Recall": recall,
        "Confusion Matrix": conf_matrix.tolist()
    }
    
    # Get filenames for saving
    if hasattr(dataset, 'get_filenames'):
        filenames = dataset.get_filenames()
    else:
        filenames = [f"Graph_{i}" for i in range(len(dataset))]
    
    excel_path = save_metrics_to_excel(metrics, dataset_info, best_params, args.output_dir, args.model_name)

    print(f"\n=== TESTING COMPLETED ===")
    print(f"Results saved to: {excel_path}")
    print(f"\n=== SUMMARY ===")
    print(f"Test Accuracy: {acc_test:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Number of Test Samples: {len(y_true)}")
    print(f"Class Distribution - ON: {class_on}, OFF: {class_off}")

    print(f"Testing completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
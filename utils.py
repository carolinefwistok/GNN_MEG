import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch


def edge_filtering(graph, top_k, threshold):
    '''
    Applies edge filtering to a single graph before inputting to the GNN model.

    INPUTS:
        - graph     : Graph object
        - top_k     : Integer defining the number of edges to keep
        - threshold : Float defining the threshold for edge weights

    OUTPUT:
        - graph     : Graph object with filtered edges
    '''
    # Retrieve edge indices and edge weights from the graph
    edge_index, edge_weight = graph.edge_index, graph.edge_attr

    # Apply threshold filtering if specified
    if threshold is not None:
        mask = edge_weight > threshold
        edge_index = edge_index[:, mask]
        edge_weight = edge_weight[mask]
        print('Num edges after threshold filtring:', edge_index.shape[1])

    # Apply top-K filtering if specified
    if top_k is not None:
        # Get the indices of the top K edges
        if edge_weight.numel() > top_k:
            top_k_indices = torch.topk(edge_weight, top_k).indices
            edge_index = edge_index[:, top_k_indices]
            edge_weight = edge_weight[top_k_indices]
            print('Num edges after top-K filtering:', edge_index.shape[1])

    # Update graph with filtered edges
    graph.edge_index = edge_index
    graph.edge_attr = edge_weight
    print('Num edges after edge filtering (total):', graph.edge_index.shape[1])

    return graph

def plot_train_results(results, best_result):
    '''
    Plots the training results by plotting the losses and accuracies of both the train and validation set.

    INPUT: N/A
    OUTPUT: N/A
    '''

    # Plot accuracies for different hyperparameters
    ax = None
    for result in results:
        label = f"lr={result.config['lr']:.5f}"
        if ax is None:
            ax = result.metrics_dataframe.plot("training_iteration", "train_accuracy", label=label)
        else:
            result.metrics_dataframe.plot("training_iteration", "train_accuracy", ax=ax, label=label)
    ax.set_title("Train Accuracy across training iterations for all LRs")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Training Accuracy")
    plt.show()

    # Plot losses and accuracies for best configuration
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(best_result.metrics_dataframe['training_iteration'], 
             best_result.metrics_dataframe['train_loss'], 
             label='Train loss')
    plt.plot(best_result.metrics_dataframe['training_iteration'], 
             best_result.metrics_dataframe['val_loss'], 
             label='Validation loss')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Training iteration')
    plt.title('Training and Validation Loss')
    plt.grid(True)
    plt.subplot(122)
    plt.plot(best_result.metrics_dataframe['training_iteration'], 
             best_result.metrics_dataframe['train_accuracy'], 
             label='Train accuracy')
    plt.plot(best_result.metrics_dataframe['training_iteration'], 
             best_result.metrics_dataframe['val_accuracy'], 
             label='Validation accuracy')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('Training iteration')
    plt.title('Training and Validation Accuracy')
    plt.grid(True)
    plt.show()

def save_results(results_df, best_params, best_result, output_directory, duration, overlap, num_graphs):
    '''
    Saves the training results and hyperparameter configurations to an Excel file.
    
    INPUTS:
        - results_df: DataFrame containing the results of the training.
        - best_params: Dictionary of the best hyperparameter configuration.
        - output_directory: Directory where the Excel file will be saved.
        - duration: Duration used for creating the dataset.
        - overlap: Overlap used for creating the dataset.
        - num_graphs: Total number of graphs created.
    
    OUTPUT: N/A
    '''

    # Create a subfolder name based on duration, overlap, and number of graphs
    subfolder_name = f'graphs_{num_graphs}_duration_{duration}_overlap_{overlap}'
    subfolder_path = os.path.join(output_directory, subfolder_name)

    # Create the subfolder if it doesn't exist
    os.makedirs(subfolder_path, exist_ok=True)

    # Get current date and time for the filename
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(subfolder_path, f'{current_time}_training_results.xlsx')

    # Create results DataFrame with specified columns
    columns_to_export = [
        'config/n_layers',
        'config/dropout_rate', 
        'config/conv1_hidden_channels',
        'config/lr',
        'config/batch_size',
        'config/weight_decay',
        'config/top_k',
        'config/threshold',
        'train_accuracy',
        'val_accuracy',
        'train_loss',
        'val_loss',
    ]
    results_df_export = results_df[columns_to_export]

    # Create a DataFrame for the best parameters
    best_params_df = pd.DataFrame([best_params])

    # Create a DataFrame for training and validation loss
    training_iter = best_result.metrics_dataframe['training_iteration']
    train_loss = best_result.metrics_dataframe['train_loss']
    val_loss = best_result.metrics_dataframe['val_loss']
    iteration_loss_df = pd.DataFrame({'iter': training_iter,
                                      'train_loss': train_loss,
                                      'val_loss': val_loss})

    # Create a Pandas Excel writer using XlsxWriter as the engine
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Write the results DataFrame to the Excel file
        results_df_export.to_excel(writer, sheet_name='Results', index=False)
        # Write the best parameters DataFrame to another sheet
        best_params_df.to_excel(writer, sheet_name='Best Parameters', index=False)
        # Write the training iteration, train loss, and val loss to another sheet
        iteration_loss_df.to_excel(writer, sheet_name='Training and validation loss', index=False)
    
    print(f'Results saved to {output_file}')

def save_metrics_to_txt(metrics, filenames, output_directory, filename):
    '''
    Saves the metrics to a CSV file.

    INPUT:
        - metrics          : Dictionary of metrics to save
        - filenames        : List of filenames
        - output_directory : Directory to save the text file
        - filename         : Name of the text file
    
    OUTPUT: N/A
    '''
    os.makedirs(output_directory, exist_ok=True)
    file_path = os.path.join(output_directory, filename)

    with open(file_path, mode='w') as file:
        file.write("\nFilenames:\n")
        for fname in filenames:
            file.write(f"{fname}\n")
        file.write("\nMetrics:\n")
        for key, value in metrics.items():
            file.write(f"{key}: {value}\n")

    print(f"Metrics saved to {file_path}")
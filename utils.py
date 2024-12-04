import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
    
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
        'config/hidden_channels',
        'config/lr',
        'config/batch_size',
        'train_accuracy',
        'val_accuracy',
        'train_loss',
        'val_loss'
    ]
    results_df_export = results_df[columns_to_export]

    # Create a DataFrame for the best parameters
    best_params_df = pd.DataFrame([best_params])

    # Create a DataFrame for training and validation loss
    training_iter = best_result.metrics_dataframe['training_iteration']
    train_loss = best_result.metrics_dataframe['train_loss']
    val_loss = best_result.metrics_dataframe['val_loss']
    iteration_loss_df = pd.DataFrame({'training_iteration': training_iter,
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
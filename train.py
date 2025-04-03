import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import GradScaler, autocast
from ray import train
from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow
import tempfile
import mlflow.pytorch
import torch.nn.functional as F
from dataset import MEGGraphs
from model import GNN
from utils import edge_filtering


def calculate_metrics(y_pred, y_true, epoch, phase):
    '''
    Calculate model performance metrics that are logged in MLflow.
    
    INPUTS: 
        - y_pred    : Predictions from the model
        - y_true    : Actual label
        - epoch     : Number of current epoch in model training
        - phase     : Specifies 'training' or 'validation' phase
    OUTPUT: N/A
    '''

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted')
    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        roc_auc = 0.0  # Handle case where ROC AUC cannot be computed

    # Log metrics
    mlflow.log_metric(f"{phase}_accuracy", accuracy, step=epoch)
    mlflow.log_metric(f"{phase}_precision", precision, step=epoch)
    mlflow.log_metric(f"{phase}_recall", recall, step=epoch)
    mlflow.log_metric(f"{phase}_roc_auc", roc_auc, step=epoch)

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

    # Set up MLflow for logging
    tracking_uri = config.pop("tracking_uri", None)
    setup_mlflow(
        config,
        tracking_uri=tracking_uri
    )

    # Start an MLflow run
    mlflow.start_run()

    # Apply edge filtering to the input graphs of the train set using top_k and threshold filtering
    top_k = config['top_k']
    threshold = config['threshold']
    filtered_graphs_train = dataset_train.copy()
    filtered_graphs_train = [edge_filtering(graph, top_k, threshold) for graph in filtered_graphs_train]
    print('graph filtered', filtered_graphs_train[0])
    print('top_k', top_k)
    print('threshold', threshold)

    # Calculate class weights
    y_train_np = np.array(y_train)
    class_counts = torch.tensor([len(y_train_np[y_train_np == 0]), len(y_train_np[y_train_np == 1])], dtype=torch.float)
    print('class counts', class_counts)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()

    # Define criterion for model training with class weights
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    # criterion = torch.nn.CrossEntropyLoss()

    # Retrieve GNN model (see model.py)
    model = GNN(
        n_layers=config['n_layers'],
        dropout_rate=config['dropout_rate'],
        conv1_hidden_channels=config['conv1_hidden_channels'], 
        conv2_hidden_channels=config['conv1_hidden_channels'],
        conv3_hidden_channels=config['conv1_hidden_channels'],
        conv4_hidden_channels=config['conv1_hidden_channels'],
        dataset=dataset,
        top_k=config['top_k'],
        threshold=config['threshold']
    )

    # Split train data in train and validation set 
    subset_train, subset_val, y_subset_train, y_subset_val = train_test_split(
        filtered_graphs_train, 
        y_train, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_train
    )

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

    # Early stopping parameters
    best_val_loss = float('inf')  # Initialize best validation loss to infinity
    patience = 10  # Number of epochs to wait before stopping
    patience_counter = 0  # Initialize patience counter
    min_epochs = 30  # Minimum number of epochs to train

    # Define a checkpoint so the updated model parameters can be retrieved at a later moment
    checkpoint = train.get_checkpoint()
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

        # Log training validation metrics to MLflow
        mlflow.log_metric("train_loss", train_loss / len(train_loader), step=epoch)
        mlflow.log_metric("train_accuracy", train_acc, step=epoch)
        mlflow.log_metric("val_loss", val_loss / len(val_loader), step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        # Adjust learning rate based on validation loss
        # scheduler.step(val_loss)

        # Log learning rate to MLflow
        # current_lr = scheduler.optimizer.param_groups[0]['lr']
        # print(f'Epoch {epoch+1}: current learning rate {current_lr}')
        # mlflow.log_metric("learning_rate", current_lr, step=epoch)

        # Save accuracies and losses together with the checkpoint
        with tempfile.TemporaryDirectory() as tempdir:
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict()},
                os.path.join(tempdir, "checkpoint.pt"),
            )

            train.report(
                {"train_accuracy": train_acc, "val_accuracy": val_acc, "train_loss": train_loss, "val_loss": val_loss, "training_iteration": epoch},
                checkpoint=train.Checkpoint.from_directory(tempdir)
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

    # Save the trained model to MLflow
    mlflow.pytorch.log_model(model, "model")

    # Save the best model's state dictionary and configuration parameters
    output_dir = os.path.join(os.path.dirname(__file__), 'Output')
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, f"best_model{model_name}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, best_model_path)
    mlflow.log_artifact(best_model_path, "best_model")


def test_best_model(best_result, dataset, filtered_dataset_test):
    '''
    Tests the trained model with best hyperparameters on the test set.

    INPUTS:
        - best_result               : Result object of results of best hyperparameter configuration
        - dataset                   : Dataset of graphs
        - filtered_dataset_test     : Dataset of filtered graphs for testing
    
    OUTPUT: 
        - acc_test          : Float of accuracy of model's predictions of test set
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
    
    # Put model in evaluation mode
    best_trained_model.eval()

    # Calculate accuracy of trained model on test set 
    correct = 0
    y_true = []
    y_pred = []
    for data in test_loader:
        out = best_trained_model(data.x, data.edge_index, data.edge_attr, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())

        # Append the true and predicted labels
        y_true.append(data.y.cpu().numpy())
        y_pred.append(pred.cpu().numpy())

    # Flatten the lists into arrays
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # Ensure they are of the same type
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    # Calculate performance metrics
    accuracy_test = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted')
    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        roc_auc = 0.0  # Handle case where ROC AUC cannot be computed

    # Print performance metrics
    print('Performance metrics of test set:')
    print('ROC AUC', roc_auc)
    print('Accuracy', accuracy_test)
    print('Precision', precision)
    print('Recall', recall)

    acc_test = correct/len(test_loader.dataset)
    print('Length dataset', len(test_loader.dataset))
    print('Length total', [data.y.size(0) for data in test_loader])

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['OFF', 'ON'], yticklabels=['OFF', 'ON'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    return acc_test, roc_auc, precision, recall, conf_matrix

def load_and_test_model(best_result, dataset, filtered_dataset_test):
    '''
    Loads the saved model and tests it on new data.

    INPUTS:
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
        dataset=dataset,
        top_k=best_result.config['top_k'],
        threshold=best_result.config['threshold']
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
    for data in test_loader:
        out = model(data.x, data.edge_index, data.edge_attr, data.batch, temperature=2)
        prob = F.softmax(out, dim=1)
        pred = prob.argmax(dim=1)
        correct += int((pred == data.y).sum())

        # Append the true and predicted labels
        y_true.append(data.y.cpu().numpy())
        y_pred.append(pred.cpu().numpy())

    # Flatten the lists into arrays
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # Ensure they are of the same type
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    # Calculate performance metrics
    accuracy_test = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted')
    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        roc_auc = 0.0  # Handle case where ROC AUC cannot be computed

    # Print performance metrics
    print('Performance metrics on test data:')
    print('ROC AUC', roc_auc)
    print('Accuracy', accuracy_test)
    print('Precision', precision)
    print('Recall', recall)

    acc_test = correct / len(test_loader.dataset)
    print('Length dataset', len(test_loader.dataset))
    print('Length total', [data.y.size(0) for data in test_loader])

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['OFF', 'ON'], yticklabels=['OFF', 'ON'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    return acc_test, roc_auc, precision, recall, conf_matrix

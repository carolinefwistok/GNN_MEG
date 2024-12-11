import torch
import os
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
import ray
from ray import train
from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow
import tempfile
import mlflow.pytorch
from model import GNN


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

def train_func(config, dataset, dataset_train, y_train):   
    '''
    Ray trainable that performs the training process for each of the hyperparameter configurations. 
    INPUTS: 
        - dataset           : Dataset of graphs
        - dataset_train     : Dataset of graphs for training 
        - y_train           : list of labels of train set 
    OUTPUT: N/A
    ''' 

    tracking_uri = config.pop("tracking_uri", None)
    setup_mlflow(
        config,
        # experiment_name="mlflow_hyperparameter_tuning",
        tracking_uri=tracking_uri,
    )

    # Retrieve GNN model (see model.py)
    model = GNN(
        hidden_channels=config['hidden_channels'], 
        dataset=dataset
    )

    # Split train data in train and validation set 
    subset_train, subset_val, y_subset_train, y_subset_val = train_test_split(
        dataset_train, 
        y_train, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_train
    )

    # Batch train and validation set 
    train_loader = DataLoader(
        subset_train, 
        config['batch_size'], 
        shuffle=True
    )
    val_loader = DataLoader(
        subset_val,
        config['batch_size'],
        shuffle=True
    )

    # Define optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = torch.nn.CrossEntropyLoss()

    # Define a checkpoint so the updated model parameters can be retrieved at a later moment
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
            model.load_state_dict(checkpoint_dict["model_state"])

    # Define how many times you want to train on the train set
    num_epochs = 100

    # with mlflow.start_run():
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

        # Log training metrics to MLflow
        mlflow.log_metric("train_loss", train_loss / len(train_loader), step=epoch)
        mlflow.log_metric("train_accuracy", train_acc, step=epoch)

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

        # Log validation metrics to MLflow
        mlflow.log_metric("val_loss", val_loss / len(val_loader), step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        # Calculate and log confusion matrix and other metrics
        calculate_metrics(y_val_pred, y_val_true, epoch, "val")

        # Save accuracies and losses together with the checkpoint
        with tempfile.TemporaryDirectory() as tempdir:
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict()},
                os.path.join(tempdir, "checkpoint.pt"),
            )

            train.report({"train_accuracy": train_acc, "val_accuracy": val_acc, "train_loss": train_loss, "val_loss": val_loss}, checkpoint=train.Checkpoint.from_directory(tempdir))
    
    # Save the trained model to MLflow
    mlflow.pytorch.log_model(model, "model")

def test_best_model(best_result, dataset, dataset_test):
    '''
    Tests the trained model with best hyperparameters on the test set. 
    INPUTS:
        - best_result       : Result object of results of best hyperparameter configuration
        - dataset           : Dataset of graphs
        - dataset_test      : Dataset of graphs for testing
    
    OUTPUT: 
        - acc_test          : Float of accuracy of model's predictions of test set
    '''
    # Initialize model with best hyperparameter for 'hidden_channels'
    best_trained_model = GNN(hidden_channels=best_result.config['hidden_channels'], dataset=dataset)

    # Retrieve trained model parameters from the checkpoint of the best result and load onto the initialized model
    with best_result.checkpoint.as_directory() as checkpoint_dir:
        state_dict = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pt'))
        best_trained_model.load_state_dict(state_dict['model_state'])

    # Batch test dataset with best batch size
    test_loader = DataLoader(
        dataset_test, 
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

    # Calculate accuracy
    accuracy_test = accuracy_score(y_true, y_pred)

    print('confusion matrix', confusion_matrix(y_true, y_pred))

    acc_test = correct/len(test_loader.dataset)

    return acc_test

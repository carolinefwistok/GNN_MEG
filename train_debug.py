#%%
import torch
import os
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import ray
# from ray import train, tune
import tempfile
from model import GNN
from main import create_dataset, split_train_test

#%%
def train_func(config, dataset, dataset_train, y_train):   
    '''
    Ray trainable that performs the training process for each of the hyperparameter configurations. 
    INPUTS: 
        - dataset           : Dataset of graphs
        - dataset_train     : Dataset of graphs for training 
        - y_train           : list of labels of train set 
    OUTPUT: N/A
    ''' 
    
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
    checkpoint = ray.train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
            model.load_state_dict(checkpoint_dict["model_state"])

    # Define how many times you want to train on the train set
    num_epochs = 100

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
        for data in train_loader:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            pred = out.argmax(dim=1)
            correct_train += int((pred == data.y).sum())

        train_acc = correct_train/len(train_loader.dataset)

        # Calculate accuracy and loss of trained model on validation set
        correct_val = 0
        val_loss = 0
        for data in val_loader:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out, data.y)
            pred = out.argmax(dim=1)
            val_loss += loss.item()
            correct_val += int((pred == data.y).sum())

        val_acc = correct_val/len(val_loader.dataset)

        # Save accuracies and losses together with the checkpoint
        with tempfile.TemporaryDirectory() as tempdir:
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict()},
                os.path.join(tempdir, "checkpoint.pt"),
            )

            ray.train.report({"train_accuracy": train_acc, "val_accuracy": val_acc, "train_loss": train_loss, "val_loss": val_loss}, checkpoint=ray.train.Checkpoint.from_directory(tempdir))
    return 'Training complete'


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
    for data in test_loader:
        out = best_trained_model(data.x, data.edge_index, data.edge_attr, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())

    acc_test = correct/len(test_loader.dataset)
    return acc_test

#%%
dataset = create_dataset()
print(dataset)

dataset_train, dataset_test, y_train, y_test = split_train_test(dataset)

#%%
# Terminate processes started by ray.init(), so you can define a local _temp_dir to store the Ray process files
ray.shutdown()

# Ensure the paths exist
os.makedirs(r"F:\MEG GNN\GNN\Ray_temp", exist_ok=True)
os.makedirs(r"F:\MEG GNN\GNN\Ray_results", exist_ok=True)

# ray.init(_temp_dir=r"F:\MEG GNN\GNN\Ray_temp")
# ray.init(address='auto', runtime_env={"working_dir": "F:\MEG GNN\GNN\Ray_temp"})
# Initialize Ray with a specified runtime environment
ray.init(
    runtime_env={
        "working_dir": r"F:\MEG GNN\GNN\Ray_temp",
        "pip": ["torch", "torch-geometric", "ray[tune]", "tensorboard"],
        "py_modules": [r"F:\MEG GNN\GNN\model.py", r"F:\MEG GNN\GNN\train.py"]  # Add Python modules for import in Ray workers
    }
)

#%%
# Make sure Ray doesn't change the working directory to the trial directory, so you can define your own (relative) path to store results 
os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"

# Make sure Ray can handle reporting more than one metric 
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

#%%
# Define path where results need to be stored
run_config = ray.train.RunConfig(
    storage_path=r"F:\MEG GNN\GNN\Ray_results"
)

# Define hyperparameter search space 
search_space = {
    'hidden_channels': ray.tune.choice([16, 32, 64, 128]),
    'lr': ray.tune.loguniform(1e-5, 1e-1),
    'batch_size': ray.tune.choice([2, 4, 6, 8])
}

# Define which metric you want Ray to base 'best_results' on, whether that metric needs to be 'max' or 'min', and how many configurations you want it to try
tune_config = ray.tune.TuneConfig(
    metric='val_accuracy',
    mode='max',
    num_samples=5
)

#%%
# Perform the training and hyperparameter tuning
tuner = ray.tune.Tuner(
    ray.tune.with_parameters(train_func, dataset=dataset, dataset_train=dataset_train, y_train=y_train),
    param_space=search_space,
    tune_config=tune_config,
    run_config=run_config
)

#%%
results = tuner.fit()

#%%
# Retrieve best result
best_result = results.get_best_result()

# Retrieve hyperparameter configuration of best result
best_params = best_result.config
# %%

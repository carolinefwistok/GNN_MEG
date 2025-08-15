import torch
import pandas as pd
import os
from torch_geometric.utils import to_networkx
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from collections import Counter, defaultdict
import scipy.io

from subgraph_x import SubgraphX
from task_enum import Task
from utils import edge_filtering

# Select your device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def calculate_explain_and_accuracies(data, best_trained_model, dataset, best_result, input_type, n_min=5):
    """
    Calculates the explanation for a specific graph.
    Also calculates accuracy difference between the normal graph and the graph with the important subgraph occluded.  

    INPUTS:
        - data:                     A PyTorch Geometric graph object of the data.  
        - best_trained_model:       The trained GNN model to perform graph classification
        - dataset:                  The graph dataset
        - best_result:              The best result of the model on the validation set
        - input_type:               Type of input data ('fif' or 'scout')
        - n_min:                    Minimum number of nodes to be considered as important subgraph (default is 5)

    OUTPUTS:
        - avg_accuracy_normal:      The average accuracy for the normal graph across all trials
        - avg_accuracy_occluded:    The average accuracy when the important subgraph is occluded
        - all_explanation_nodes:    List of important subgraphs (nodes) identified across iterations
        - all_explanation_labels:   Labels associated with the identified subgraphs
        - most_common_nodes:        Most frequently identified important nodes across iterations
        - most_common_labels:       Most frequently associated labels for these nodes
        - accuracy_difference:      Difference between average accuracies of the normal and occluded graphs
        - node_frequencies:         Frequency distribution of identified important nodes
        - avg_fidelity_ratio:       Average fidelity ratio, indicating how much the subgraph contributes to model prediction confidence
    """
   
    # Initialize variables
    correct_normal = 0
    correct_occluded = 0
    all_explanation_nodes = []

    # Attain most important subgraph
    explainer = SubgraphX(best_trained_model, best_result.config['n_layers'], exp_weight=0.5, m=100, t=10, task = Task.GRAPH_CLASSIFICATION)
    explanation_nodes, explanation_scores = explainer(data, n_min=n_min)

    # Save important nodes to list
    all_explanation_nodes.extend(explanation_nodes)  

    # Compute accuracy of model on normal graph
    best_trained_model.eval()
    scores_normal = best_trained_model(data.x, data.edge_index, data.edge_attr, data.batch, temperature=2)
    prob_normal = F.softmax(scores_normal, dim=1)
    predicted_normal = torch.argmax(prob_normal, dim=-1)
    correct_normal += int((predicted_normal == data.y).sum())
    logit_normal = scores_normal[torch.arange(scores_normal.size(0)), predicted_normal]

    # Compute accuracy of model with occluded subgraph, by erasing node features of important nodes
    node_tensor = torch.tensor(list(explanation_nodes)).long()
    x_occluded = torch.clone(data.x)
    x_occluded[node_tensor, :] = 0  
    scores_occluded = best_trained_model(x_occluded, data.edge_index, data.edge_attr, data.batch, temperature=2)
    prob_occluded = F.softmax(scores_occluded, dim=1)
    predicted_occluded = torch.argmax(prob_occluded, dim=-1)
    correct_occluded += int((predicted_occluded == data.y).sum())
    logit_occluded = scores_occluded[torch.arange(scores_occluded.size(0)), predicted_normal]

    # Compute fidelity score
    fidelity_score = torch.sigmoid(logit_normal - logit_occluded).mean().item()

    # Compute difference in fidelity ratio between normal and occluded graph
    scores_normal = scores_normal[0, predicted_normal]
    scores_occluded = scores_occluded[0, predicted_normal]
    ratio = -1*((scores_occluded - scores_normal)/(abs(scores_occluded)+abs(scores_normal))).item()

    # Calculate averages over number of iterations
    avg_accuracy_normal = correct_normal
    avg_accuracy_occluded = correct_occluded
    accuracy_difference = avg_accuracy_normal-avg_accuracy_occluded
    avg_fidelity_ratio = np.mean(ratio)
    avg_fidelity_score = fidelity_score

    # Count the frequency of all nodes
    node_counts = Counter(all_explanation_nodes)

    # Find the most common nodes and their frequencies
    most_common_nodes, node_frequencies = zip(*node_counts.most_common(n_min)) 

    # Get the labels of the important nodes and all nodes
    if input_type == 'fif':
        raw = dataset.load_raw_data(dataset.raw_paths[0])
        channels = raw.info["ch_names"]
        nodes_dict = {n: channel for n, channel in enumerate(channels)}
    elif input_type == 'scout':
        mat = scipy.io.loadmat("/scratch/cwitstok/Data/Scout_all/Mat_files/PT01_BURST_run07_SCOUTS.mat")
        descriptions = mat['Description'][:]
        cleaned_descriptions = [str(desc[0]).strip("'[]'") for desc in descriptions]
        nodes_dict = {n: desc for n, desc in enumerate(cleaned_descriptions)}
        nodes_dict = adjust_scout_names(nodes_dict)
    
    most_common_labels = [nodes_dict[node_idx] for node_idx in most_common_nodes]

    # Print the most common nodes and their frequencies
    for label, freq in zip(most_common_labels, node_frequencies):
        print(f"Label: {label}, Frequency: {freq}")

    return avg_accuracy_normal, avg_accuracy_occluded, most_common_nodes, most_common_labels, accuracy_difference, node_frequencies, avg_fidelity_ratio, avg_fidelity_score

def sparsity(data, most_common_nodes):
    '''
    Gives sparsity given a number of nodes used for explanation.

    INPUTS:
        - data :                A PyTorch Geometric graph object of the data.
        - most_common_nodes :   The subgraph consisting of important nodes for prediction
    OUTPUT: 
        - sparsity_score :      Fraction of nodes needed for explanation compared to total amount of nodes (close to 1 = small amount of nodes, close to 0 = almost all of the nodes)
    '''
    # Calculate sparsity score
    sparsity_score = 1 - (len(most_common_nodes) / data.num_nodes)
    return sparsity_score

def excel_explanation(excel_path, target_index, most_common_nodes, most_common_labels, node_frequencies, avg_fidelity_score, avg_fidelity_ratio, sparsity_score, avg_accuracy_normal, avg_accuracy_occluded, accuracy_difference, run_time, n_min=5):
    '''
    Logs all outcomes of explanability to an Excel file.

    INPUTS:
        - excel_path :              Specific path to your Excel file
        - target_index :            Specific index of graph in the test files
        - most_common_nodes :       Most frequently identified important nodes across iterations
        - most_common_labels :      Most frequently associated labels for these nodes
        - node_frequencies :        Frequency distribution of identified important nodes
        - avg_fidelity_score :      Average fidelity score across iterations
        - avg_fidelity_ratio :      Average fidelity ratio, indicating how much the subgraph contributes to model prediction confidence
        - sparsity_score :          Fraction of nodes needed for explanation compared to total amount of nodes (close to 1 = small amount of nodes, close to 0 = almost all of the nodes)
        - avg_accuracy_normal :     The average accuracy for the normal graph across all trials
        - avg_accuracy_occluded :   The average accuracy when the important subgraph is occluded
        - accuracy_difference :     Difference between average accuracies of the normal and occluded graphs
        - run_time :                Amount of time needed to run code section in minutes
        - n_min :                   Minimum number of nodes to consider for explanation
    OUTPUT: N/A
    '''

    # Check if the Excel file already exists
    if os.path.exists(excel_path):
        # Load the existing file
        df = pd.read_excel(excel_path)
    else:
        # Create a new DataFrame if the file doesn't exist
        df = pd.DataFrame(columns=["Train Index", "Most common Nodes", "Most common Labels", "Node frequencies", "Fidelity Score", "Mean fidelity ratio", "Sparsity Score", "Normal accuracy", "Occluded accuracy", "Accuracy difference", "Run Time"])

    # Check if the graph index already exists in the Excel file
    if target_index in df["Train Index"].values:
        # Update the existing row with the same graph index
        row_index = df.index[df["Train Index"] == target_index].tolist()[0]  
        df.loc[row_index, "Most common Nodes"] = ", ".join(map(str, most_common_nodes))
        df.loc[row_index, "Most common Labels"] = ", ".join(most_common_labels)
        df.loc[row_index, "Node frequencies"] = ", ".join(map(str, node_frequencies))
        df.loc[row_index, "Fidelity Score"] = avg_fidelity_score
        df.loc[row_index, "Sparsity Score"] = sparsity_score
        df.loc[row_index, "Mean fidelity ratio"] = avg_fidelity_ratio
        df.loc[row_index, "Normal accuracy"] = avg_accuracy_normal
        df.loc[row_index, "Occluded accuracy"] = avg_accuracy_occluded
        df.loc[row_index, "Accuracy difference"] = accuracy_difference
        df.loc[row_index, "Run Time"] = run_time

    else:
        # Add a new row if the graph index doesn't exist
        new_row = {
            "Train Index": target_index,
            "Most common Nodes": ", ".join(map(str, most_common_nodes)),
            "Most common Labels": ", ".join(most_common_labels),
            "Node frequencies": ", ".join(map(str, node_frequencies)),
            "Fidelity Score": avg_fidelity_score,
            "Sparsity Score": sparsity_score,
            "Mean fidelity ratio": avg_fidelity_ratio,
            "Normal accuracy": avg_accuracy_normal,
            "Occluded accuracy": avg_accuracy_occluded,
            "Accuracy difference": accuracy_difference,
            "Run Time": run_time
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Add a summary row at the bottom
    if "Total" in df["Train Index"].values:
        # Remove the existing "Total" row if it exists
        df = df[df["Train Index"] != "Total"]

    # Aggregate all labels and their frequencies
    all_labels = []
    all_frequencies = []
    all_nodes = []  # To store all node indices
    for labels, frequencies, nodes in zip(df["Most common Labels"], df["Node frequencies"], df["Most common Nodes"]):
        if pd.notna(labels) and pd.notna(frequencies) and pd.notna(nodes):
            label_list = labels.split(", ")
            frequency_list = map(int, frequencies.split(", "))
            node_list = map(int, nodes.split(", "))  # Convert node indices to integers
            all_labels.extend(label_list)
            all_frequencies.extend(frequency_list)
            all_nodes.extend(node_list)

    # Count the frequency of each label
    label_counts = Counter(all_labels)
    label_frequencies = [freq for _, freq in label_counts.most_common(n_min)]

    # Count the frequency of each node
    node_counts = Counter(all_nodes)
    most_common_nodes_str = [node for node, _ in node_counts.most_common(n_min)]

    # Format the most common labels and their frequencies
    # most_common_labels_str = ", ".join([f"{label} ({freq})" for label, freq in most_common_labels])

    # Compute averages or totals for numeric columns
    summary_row = {
        "Train Index": "Total",
        "Most common Nodes": most_common_nodes,
        "Most common Labels": most_common_labels,
        "Node frequencies": label_frequencies,
        "Fidelity Score": df["Fidelity Score"].mean(),
        "Mean fidelity ratio": df["Mean fidelity ratio"].mean(),
        "Sparsity Score": df["Sparsity Score"].mean(),
        "Normal accuracy": df["Normal accuracy"].mean(),
        "Occluded accuracy": df["Occluded accuracy"].mean(),
        "Accuracy difference": df["Accuracy difference"].mean(),
        "Run Time": df["Run Time"].sum()
    }
    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

    # Write the DataFrame back to the Excel file
    df.to_excel(excel_path, index=False)

    # Print if successful
    print(f"Data for graph {target_index} saved to {excel_path}")
    return most_common_nodes

def visualize_graph_with_subgraph(data, label, explanation_nodes, dataset, input_type, explain_dir, target_index, edge_weights):
    '''
    Visualizes the network of one of the graphs in the dataset, highlighting the important nodes and saving the plot to the output folder.

    INPUTS:
        - data :                A PyTorch Geometric graph object of the data.
        - label :               Label of the graph
        - explanation_nodes :   The subgraph consisting of important nodes for prediction
        - dataset :             The graph dataset
        - input_type :          Type of input data ('fif' or 'scout')
        - explain_dir :         Directory to save the explanation plot
        - target_index :        Specific index of graph in train dataset
        - edge_weights :        Dictionary of edge weights (optional, default=None)
    OUTPUT: Saved plot in the "Explainability output" folder.
    '''

    if input_type == 'fif':
        # Retrieve the raw data (if necessary for other visual elements)
        raw = dataset.load_raw_data(dataset.raw_paths[0])

        # Retrieve the channel names and positions (if needed)
        channels = raw.info["ch_names"]
        node_labels_dict = {n: channel for n, channel in enumerate(channels)}
        ch_pos = [ch['loc'][:2] for ch in raw.info['chs']]
        node_positions_dict = {idx: pos for idx, pos in enumerate(ch_pos)}
        print('Node positions:', node_positions_dict)
        adjusted_node_labels_dict = node_labels_dict.copy()

    elif input_type == 'scout':
        mat = scipy.io.loadmat("/scratch/cwitstok/Data/Scout_all/Mat_files/PT01_BURST_run07_SCOUTS.mat")
        descriptions = mat['Description'][:]
        cleaned_descriptions = [str(desc[0]).strip("'[]'") for desc in descriptions]

        # Load the scout coordinates from the CSV file
        scout_coords_file = '/scratch/cwitstok/Data/source_scouts_coordinates.csv'
        scout_coords_df = pd.read_csv(scout_coords_file)

        # Create a dictionary for scout positions
        scout_positions = {}
        for idx, scout in enumerate(cleaned_descriptions):
            # Find the row in the CSV file corresponding to the current scout
            scout_row = scout_coords_df[scout_coords_df['Scout'] == scout]
    
            if not scout_row.empty:
                # Extract the scout name and coordinates
                scout_name = scout_row['Scout'].iloc[0]  # Extract the scalar value
                x, y = scout_row['x_2D'].iloc[0], scout_row['y_2D'].iloc[0]
                scout_positions[scout_name] = (x, y)
            else:
                # Assign random coordinates if the scout is not found in the CSV file
                scout_positions[scout] = np.random.rand(2)

        # Convert node positions and labels into dictionaries
        node_positions_dict = {i: pos for i, pos in enumerate(scout_positions.values())}
        node_labels_dict = {i: label for i, label in enumerate(scout_positions.keys())}
        adjusted_node_labels_dict = adjust_scout_names(node_labels_dict)

    # Convert PyG data to a NetworkX graph
    g = to_networkx(data, to_undirected=True)

    # Ensure all nodes have positions
    all_nodes = list(g.nodes)
    fixed_node_positions = {
        node: node_positions_dict.get(node, (0, 0)) for node in all_nodes
    }

    # Define node sizes for normal and important nodes
    normal_node_size = 300
    important_node_size = 550  # Size for important nodes (subgraph)

    # Extract edge weights for visualization
    if edge_weights is None:
        edge_weights = {edge: 1.0 for edge in g.edges()}  # Default weight of 1.0 for all edges

    edge_weights_list = [edge_weights.get((u, v), edge_weights.get((v, u), 1.0)) for u, v in g.edges()]

    # Normalize edge weights for consistent visualization
    min_weight, max_weight = min(edge_weights_list), max(edge_weights_list)
    normalized_weights = [(w - min_weight) / (max_weight - min_weight) if max_weight > min_weight else 1.0 for w in edge_weights_list]

    # Visualize the graph
    plt.figure(figsize=(12, 8))

    # Draw the whole graph with normal nodes
    nx.draw_networkx(
        g,
        pos=node_positions_dict,
        with_labels=True,
        node_color='lightblue',
        edge_color='k',  # Use normalized weights for edge colors
        node_size=normal_node_size,
        width=[w * 5 for w in normalized_weights]  # Scale edge widths
    )

    # Highlight the important subgraph nodes by drawing them larger and in a different color
    nx.draw_networkx_nodes(
        g,
        pos=node_positions_dict,
        nodelist=explanation_nodes,
        node_color='red',
        node_size=important_node_size,
    )

    # labels = nx.draw_networkx_labels(
    #     g,
    #     pos=node_positions_dict,
    #     labels=node_labels_dict,
    #     font_color='white',
    #     font_weight='bold',
    #     font_size=9,
    #     bbox=dict(facecolor='none', edgecolor='none')
    # )
    # # Apply path effects to each label for the black border
    # for graph_label in labels.values():
    #     graph_label.set_path_effects([
    #         path_effects.Stroke(linewidth=1.5, foreground='black'),  # Black border
    #         path_effects.Normal()  # Normal white text
    #     ])

    # Add a legend for the most important nodes
    most_important_labels = {node: adjusted_node_labels_dict[node] for node in explanation_nodes if node in adjusted_node_labels_dict}
    legend_labels = [f"Node {node}: {label}" for node, label in most_important_labels.items()]
    legend_text = "\n".join(legend_labels)
    legend_text = "Nodes in Subgraph\n" + legend_text
    plt.gca().legend(
        handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Important Nodes')],
        loc='upper right',
    )
    plt.text(
        1.05, 0.5, legend_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5)
    )

    # Save the plot to the "Explainability output" folder
    output_path = os.path.join(explain_dir, f"graph_{target_index}_explanation.png")
    if label == 0:
        plt.title(f"Graph {target_index} (stimulation OFF) with highlighted Subgraph")
    elif label == 1:
        plt.title(f"Graph {target_index} (stimulation ON) with highlighted Subgraph")
    elif label == 'total':
        plt.title(f"Graph with highlighted Subgraph")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"Graph visualization saved to {output_path}")

def extract_edge_weights(graph):
    '''
    Extracts edge weights from a PyTorch Geometric graph and returns them as a dictionary.

    INPUT:
        - graph:        A PyTorch Geometric graph object of the data.
    OUTPUT:
        - edge_weights: A dictionary where keys are edge tuples (source, target) and values are edge weights.
    '''

    # Convert edge_index and edge_attr to numpy arrays for easier processing
    edge_index = graph.edge_index.numpy()  # Shape: [2, num_edges]
    edge_attr = graph.edge_attr.numpy()   # Shape: [num_edges]

    # Create a dictionary of edge weights
    edge_weights = {}
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        edge_weights[(src, dst)] = edge_attr[i]

    return edge_weights

def average_edge_weights(edge_weights_graph):
    '''
    Computes the average edge weights from a list of edge weight dictionaries.

    INPUT:
        - edge_weights_graph:   A list of dictionaries where each dictionary contains edge weights for a graph.
    OUTPUT:
        - avg_edge_weights:     A dictionary where keys are edge tuples (source, target) and values are the average edge weights.
    '''

    # Initialize dictionaries to store sums and counts of edge weights
    edge_sums = defaultdict(float)
    edge_counts = defaultdict(int)

    # Iterate through each graph's edge weights and accumulate sums and counts
    for edge_weights in edge_weights_graph:
        for edge, weight in edge_weights.items():
            edge_sums[edge] += weight
            edge_counts[edge] += 1

    # Calculate average edge weights
    avg_edge_weights = {edge: edge_sums[edge] / edge_counts[edge] for edge in edge_sums}
    return avg_edge_weights

def adjust_scout_names(node_labels_dict):
    '''
    Adjusts the node labels in the dictionary to match the desired format.

    INPUT:
        - node_labels_dict:             Dictionary containing node labels.
    OUTPUT:
        - adjusted_node_labels_dict:    Dictionary with adjusted node labels.
    '''

    # Adjust node labels using the mapping
    scout_name_mapping = {
        'S1 hand left': 'S1 Hand L',
        'S1 foot left': 'S1 Foot L',
        'S2 left': 'S2 L',
        'Insula anterior left': 'Insula ant. L',
        'S1 foot right': 'S1 Foot R',
        'Insula posterior left': 'Insula post. L',
        'M1 left': 'M1 L',
        'SMA left': 'SMA L',
        'Sup parietal lobule left': 'sup. Parietal L',
        'Sup frontaal left': 'sup. Frontal L',
        'DLPFC left': 'DLPFC L',
        'Parietaal post left': 'post. Parietal L',
        'Occipitaal sup left': 'sup. Occipital L',
        'Occipitaal left': 'Occipital L',
        'Precentraal left': 'Precentral L',
        'DLPFC right': 'DLPFC R',
        'Precentraal right': 'Precentral R',
        'Occipitaal sup right': 'sup. Occipital R',
        'Occipitaal right': 'Occipital R',
        'Parietaal post right': 'post. Parietal R',
        'Sup frontaal right': 'sup. Frontal R',
        'S1 hand right': 'S1 Hand R',
        'S2 right': 'S2 R',
        'SMA right': 'SMA R',
        'M1 right': 'M1 R',
        'PCC': 'PCC',
        'MCC': 'MCC',
        'Insula anterior right': 'ant. Insula R',
        'Sup parietal right': 'sup. Parietal R',
        'Insula posterior right': 'post. Insula R',
        'OFC right': 'OFC R',
        'ACC': 'ACC',
        'OFC left': 'OFC L'
    }
    adjusted_node_labels_dict = {i: scout_name_mapping.get(label, label) for i, label in node_labels_dict.items()}
    return adjusted_node_labels_dict
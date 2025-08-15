import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from collections import defaultdict
import pandas as pd
import scipy.io
import networkx as nx
import matplotlib.patheffects as path_effects
from torch_geometric.utils import to_networkx
import re
from torch_geometric.data import Data

def load_scout_names():
    '''
    Loads the scout names and coordinates.
    '''
    # Load the scout names from the .mat file
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
            scout_name = scout_row['Scout'].iloc[0]
            x, y = scout_row['x_2D'].iloc[0], scout_row['y_2D'].iloc[0]
            scout_positions[scout_name] = (x, y)
        else:
            # Assign random coordinates if the scout is not found in the CSV file
            print(f"Warning: Scout '{scout}' not found in coordinates file")
            scout_positions[scout] = np.random.rand(2)

    return scout_positions, scout_coords_df, cleaned_descriptions

def load_graphs_from_directory(processed_dir):
    """
    Load all graph files from the processed directory and organize by patient and condition.
    """
    graphs_by_file = defaultdict(list)
    all_graphs = []
    
    # Get all .pt files in the directory
    for filename in os.listdir(processed_dir):
        if filename.startswith('graph_') and filename.endswith('.pt'):
            try:
                # Parse filename to extract file identifier
                parts = filename.replace('graph_', '').replace('.pt', '').split('_')
                if len(parts) >= 3:
                    patient_code = parts[0]
                    stim_type = parts[1]
                    epoch_idx = int(parts[2])
                    
                    file_identifier = f"{patient_code}_{stim_type}"
                    
                    # Load the graph
                    graph_path = os.path.join(processed_dir, filename)
                    with torch.serialization.safe_globals([Data]):
                        graph = torch.load(graph_path, weights_only=False)
                    
                    # Store with epoch index for sorting
                    graphs_by_file[file_identifier].append((epoch_idx, graph))
                    all_graphs.append(graph)
                    
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
    
    # Sort graphs by epoch index
    for file_id in graphs_by_file:
        graphs_by_file[file_id].sort(key=lambda x: x[0])
        graphs_by_file[file_id] = [graph for _, graph in graphs_by_file[file_id]]
    
    return graphs_by_file, all_graphs

def plot_connectivity_matrices_comprehensive(all_graphs, output_dir, node_labels=None):
    """
    Create comprehensive connectivity matrix analysis:
    1. Average connectivity for ON graphs
    2. Average connectivity for OFF graphs  
    3. Difference matrix (OFF - ON)
    """
    connectivity_matrices_on = []
    connectivity_matrices_off = []

    print("Computing connectivity matrices...")
    
    for graph in all_graphs:
        if graph.y is not None and graph.edge_index is not None and graph.edge_attr is not None:
            edge_index = graph.edge_index.numpy()
            edge_weight = graph.edge_attr.numpy()
            num_nodes = graph.x.shape[0]
            conn_matrix = np.zeros((num_nodes, num_nodes))
            
            # Build connectivity matrix
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[:, i]
                conn_matrix[src, dst] = edge_weight[i]
                conn_matrix[dst, src] = edge_weight[i]  # Make symmetric
            
            if graph.y.item() == 1:  # Stimulation ON
                connectivity_matrices_on.append(conn_matrix)
            else:  # Stimulation OFF
                connectivity_matrices_off.append(conn_matrix)

    print(f"Found {len(connectivity_matrices_on)} ON matrices and {len(connectivity_matrices_off)} OFF matrices")
    
    # Average connectivity matrix for ON condition
    if connectivity_matrices_on:
        avg_conn_on = np.mean(connectivity_matrices_on, axis=0)
        
        # Create figure with proper size for all labels
        plt.figure(figsize=(16, 14))
        im = plt.imshow(avg_conn_on, cmap='viridis', interpolation='nearest')
        
        # Create colorbar with proper sizing
        cbar = plt.colorbar(im, label='Edge Weight', shrink=0.8)
        cbar.ax.tick_params(labelsize=12)
        
        plt.title(f'Average Connectivity Matrix - Stimulation ON\n(n={len(connectivity_matrices_on)} graphs)', 
                 fontsize=18, fontweight='bold', pad=20)
        
        # Set all node labels if available
        if node_labels and len(node_labels) > 0:
            num_nodes = min(len(node_labels), avg_conn_on.shape[0])
            tick_positions = np.arange(num_nodes)
            
            # Set x-axis labels
            plt.xticks(tick_positions, node_labels[:num_nodes], 
                      rotation=45, ha='right', fontsize=8)
            
            # Set y-axis labels  
            plt.yticks(tick_positions, node_labels[:num_nodes], fontsize=8)
            
            print(f"Applied {num_nodes} node labels to connectivity matrix")
        else:
            # Fallback to node indices
            num_nodes = avg_conn_on.shape[0]
            tick_positions = np.arange(0, num_nodes, max(1, num_nodes // 20))
            plt.xticks(tick_positions, [f'Node {i}' for i in tick_positions], 
                      rotation=45, ha='right', fontsize=8)
            plt.yticks(tick_positions, [f'Node {i}' for i in tick_positions], fontsize=8)
        
        plt.xlabel('Brain Regions', fontsize=14, fontweight='bold')
        plt.ylabel('Brain Regions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        on_path = os.path.join(output_dir, 'connectivity_matrix_stimulation_ON.png')
        plt.savefig(on_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Stimulation ON connectivity matrix saved: {on_path}")

    # Average connectivity matrix for OFF condition
    if connectivity_matrices_off:
        avg_conn_off = np.mean(connectivity_matrices_off, axis=0)
        
        # Create figure with proper size for all labels
        plt.figure(figsize=(16, 14))
        im = plt.imshow(avg_conn_off, cmap='viridis', interpolation='nearest')
        
        # Create colorbar with proper sizing
        cbar = plt.colorbar(im, label='Edge Weight', shrink=0.8)
        cbar.ax.tick_params(labelsize=12)
        
        plt.title(f'Average Connectivity Matrix - Stimulation OFF\n(n={len(connectivity_matrices_off)} graphs)', 
                 fontsize=18, fontweight='bold', pad=20)
        
        # Set all node labels if available
        if node_labels and len(node_labels) > 0:
            num_nodes = min(len(node_labels), avg_conn_off.shape[0])
            tick_positions = np.arange(num_nodes)
            
            # Set x-axis labels
            plt.xticks(tick_positions, node_labels[:num_nodes], 
                      rotation=45, ha='right', fontsize=8)
            
            # Set y-axis labels
            plt.yticks(tick_positions, node_labels[:num_nodes], fontsize=8)
            
            print(f"Applied {num_nodes} node labels to connectivity matrix")
        else:
            # Fallback to node indices
            num_nodes = avg_conn_off.shape[0]
            tick_positions = np.arange(0, num_nodes, max(1, num_nodes // 20))
            plt.xticks(tick_positions, [f'Node {i}' for i in tick_positions], 
                      rotation=45, ha='right', fontsize=8)
            plt.yticks(tick_positions, [f'Node {i}' for i in tick_positions], fontsize=8)
        
        plt.xlabel('Brain Regions', fontsize=14, fontweight='bold')
        plt.ylabel('Brain Regions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        off_path = os.path.join(output_dir, 'connectivity_matrix_stimulation_OFF.png')
        plt.savefig(off_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Stimulation OFF connectivity matrix saved: {off_path}")

    # Difference matrix (OFF - ON)
    if connectivity_matrices_on and connectivity_matrices_off:
        avg_conn_on = np.mean(connectivity_matrices_on, axis=0)
        avg_conn_off = np.mean(connectivity_matrices_off, axis=0)
        diff_matrix = avg_conn_off - avg_conn_on
        
        abs_max = np.max(np.abs(diff_matrix))
        
        # Create figure with proper size for all labels
        plt.figure(figsize=(16, 14))
        im = plt.imshow(diff_matrix, cmap='seismic', interpolation='nearest', 
                       vmin=-abs_max, vmax=abs_max)
        
        # Create colorbar with proper sizing
        cbar = plt.colorbar(im, label='Connectivity Difference (OFF - ON)', shrink=0.8)
        cbar.ax.tick_params(labelsize=12)
        
        plt.title('Connectivity Matrix Difference (OFF - ON)\nAll Graphs', 
                 fontsize=18, fontweight='bold', pad=20)
        
        # Set all node labels if available
        if node_labels and len(node_labels) > 0:
            num_nodes = min(len(node_labels), diff_matrix.shape[0])
            tick_positions = np.arange(num_nodes)
            
            # Set x-axis labels
            plt.xticks(tick_positions, node_labels[:num_nodes], 
                      rotation=45, ha='right', fontsize=8)
            
            # Set y-axis labels
            plt.yticks(tick_positions, node_labels[:num_nodes], fontsize=8)
            
            print(f"Applied {num_nodes} node labels to difference matrix")
        else:
            # Fallback to node indices
            num_nodes = diff_matrix.shape[0]
            tick_positions = np.arange(0, num_nodes, max(1, num_nodes // 20))
            plt.xticks(tick_positions, [f'Node {i}' for i in tick_positions], 
                      rotation=45, ha='right', fontsize=8)
            plt.yticks(tick_positions, [f'Node {i}' for i in tick_positions], fontsize=8)
        
        plt.xlabel('Brain Regions', fontsize=14, fontweight='bold')
        plt.ylabel('Brain Regions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        diff_path = os.path.join(output_dir, 'connectivity_matrix_difference.png')
        plt.savefig(diff_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Connectivity difference matrix saved: {diff_path}")
    
    return avg_conn_on if connectivity_matrices_on else None, avg_conn_off if connectivity_matrices_off else None

def plot_specific_nodes_raw_features(all_graphs, output_dir, node_labels=None, fmax=100):
    """
    Plot raw node features (directly from graphs) for S2 L, S2 R, and MCC.
    Similar to plot_selected_node_features in plot_graph_analysis.py
    """
    valid_graphs = [graph for graph in all_graphs if graph.x is not None and graph.y is not None]
    
    if not valid_graphs:
        print("No valid graphs found for node features plot")
        return
    
    # Find indices for specific nodes (similar to plot_graph_analysis approach)
    target_nodes = {
        'S2 L': ['S2 L', 'S2_L', 'S2 left', 'S2_left'],
        'S2 R': ['S2 R', 'S2_R', 'S2 right', 'S2_right'], 
        'MCC': ['MCC', 'midcingulate', 'mid cingulate']
    }
    
    found_nodes = {}
    
    if node_labels:
        for target_name, search_terms in target_nodes.items():
            for idx, label in enumerate(node_labels):
                if any(term.lower() in label.lower() for term in search_terms):
                    found_nodes[target_name] = {'index': idx, 'label': label}
                    print(f"Found {target_name}: {label} at index {idx}")
                    break
    
    # Use default indices if not found (similar to plot_graph_analysis)
    if not found_nodes:
        found_nodes = {
            'S2 L': {'index': 2, 'label': node_labels[2] if node_labels and len(node_labels) > 2 else 'Node 2'},
            'S2 R': {'index': 22, 'label': node_labels[22] if node_labels and len(node_labels) > 22 else 'Node 22'},
            'MCC': {'index': 26, 'label': node_labels[26] if node_labels and len(node_labels) > 26 else 'Node 26'}
        }
        print("Using default node indices: 2, 22, 26")
    
    # Separate graphs by condition and extract node features
    node_features_on = {name: [] for name in found_nodes.keys()}
    node_features_off = {name: [] for name in found_nodes.keys()}
    
    for graph in valid_graphs:
        condition = 'on' if graph.y.item() == 1 else 'off'
        
        for node_name, node_info in found_nodes.items():
            node_idx = node_info['index']
            if node_idx < graph.x.shape[0]:  # Check if node exists in this graph
                node_feature = graph.x[node_idx].numpy()  # Raw node features
                
                if condition == 'on':
                    node_features_on[node_name].append(node_feature)
                else:
                    node_features_off[node_name].append(node_feature)
    
    # Create frequency axis
    n_freqs = len(node_features_on[list(found_nodes.keys())[0]][0]) if node_features_on[list(found_nodes.keys())[0]] else len(node_features_off[list(found_nodes.keys())[0]][0])
    frequencies = np.linspace(1, fmax, n_freqs)
    
    # Create plots for each node (similar to plot_graph_analysis style)
    plt.figure(figsize=(15, 10))
    num_nodes = len(found_nodes)
    
    for i, (node_name, node_info) in enumerate(found_nodes.items()):
        plt.subplot(num_nodes, 1, i + 1)
        
        # Plot ON condition
        if node_features_on[node_name]:
            features_on = np.array(node_features_on[node_name])
            mean_on = np.mean(features_on, axis=0)
            std_on = np.std(features_on, axis=0)
            
            plt.plot(frequencies, mean_on, color='#e41a1c', linewidth=2, 
                    label=f'Stimulation ON (n={len(features_on)})')
            plt.fill_between(frequencies, mean_on - std_on, mean_on + std_on, 
                           color='#e41a1c', alpha=0.2)
        
        # Plot OFF condition
        if node_features_off[node_name]:
            features_off = np.array(node_features_off[node_name])
            mean_off = np.mean(features_off, axis=0)
            std_off = np.std(features_off, axis=0)
            
            plt.plot(frequencies, mean_off, color='#377eb8', linewidth=2, 
                    label=f'Stimulation OFF (n={len(features_off)})')
            plt.fill_between(frequencies, mean_off - std_off, mean_off + std_off, 
                           color='#377eb8', alpha=0.2)
        
        # Customize subplot (similar to plot_graph_analysis style)
        plt.title(f'Node: {node_info["label"]}', fontsize=15)
        plt.xlabel('Frequency (Hz)', fontsize=14)
        plt.ylabel('Node features (AU)', fontsize=14)
        plt.tick_params(axis='both', labelsize=13)
        plt.xlim(0, fmax)
        plt.ylim(-0.25, 0.25)
        plt.legend(fontsize=12, loc='upper right')
        plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    features_path = os.path.join(output_dir, 'selected_node_features.png')
    plt.savefig(features_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Raw node features plot saved: {features_path}")

def plot_graph_structure_visualization(all_graphs, output_dir, node_positions, node_labels=None):
    """
    Create visualization of actual graph structures.
    Similar to plot_filtered_average_graph_with_highlight in plot_graph_analysis.py
    """
    valid_graphs = [graph for graph in all_graphs if graph.x is not None and graph.y is not None]
    
    if not valid_graphs:
        print("No valid graphs for structure visualization")
        return
    
    # Collect edge weights similar to plot_graph_analysis approach
    edge_weights_collection = {}
    
    for graph in valid_graphs:
        if graph.edge_index is not None and graph.edge_attr is not None:
            edge_index = graph.edge_index.numpy()
            edge_weight = graph.edge_attr.numpy()
            
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[:, i]
                edge_key = tuple(sorted([src, dst]))
                
                if edge_key not in edge_weights_collection:
                    edge_weights_collection[edge_key] = []
                edge_weights_collection[edge_key].append(edge_weight[i])
    
    # Average edge weights and apply threshold
    averaged_edges = {}
    for edge, weights in edge_weights_collection.items():
        avg_weight = np.mean(weights)
        averaged_edges[edge] = avg_weight
    
    # Create NetworkX graph
    G = nx.Graph()
    for (src, dst), weight in averaged_edges.items():
        G.add_edge(src, dst, weight=weight)
    
    # Add all nodes to ensure they are included
    if node_labels:
        for i in range(len(node_labels)):
            if i not in G.nodes:
                G.add_node(i)
    
    # Prepare positions
    if node_positions and node_labels:
        node_positions_list = [node_positions.get(label, (0, 0)) for label in node_labels]
        pos = {i: node_positions_list[i] for i in range(min(len(node_positions_list), len(node_labels)))}
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Get edge weights for visualization
    if G.edges():
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        min_weight, max_weight = min(edge_weights), max(edge_weights)
        if max_weight > min_weight:
            normalized_weights = [(w - min_weight) / (max_weight - min_weight) * 4 + 0.5 for w in edge_weights]
        else:
            normalized_weights = [2.0] * len(edge_weights)
    else:
        normalized_weights = []
    
    # Plot the graph (similar to plot_graph_analysis style)
    plt.figure(figsize=(12, 8))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='r', node_size=500, alpha=0.8)
    
    # Draw edges
    if G.edges():
        nx.draw_networkx_edges(G, pos, width=normalized_weights, alpha=0.6, edge_color='k')
    
    # Add labels with effects (similar to plot_graph_analysis)
    if node_labels:
        labels = {i: node_labels[i] for i in range(min(len(node_labels), len(pos)))}
        label_objects = nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, 
                                               font_weight='bold', font_color='black')
        
        # Add white outline to labels
        for label in label_objects.values():
            label.set_path_effects([
                path_effects.Stroke(linewidth=1.5, foreground='white'),
                path_effects.Normal()
            ])
    
    plt.title(f'Average Graph Structure\n(Edges: {len(G.edges())})', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # Save plot
    viz_path = os.path.join(output_dir, 'graph_structure_visualization.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Graph structure visualization saved: {viz_path}")

def main():
    """
    Main function similar to plot_graph_analysis.py structure.
    """
    parser = argparse.ArgumentParser(description="Comprehensive MEG Graph Analysis")
    parser.add_argument("--processed_dir", type=str, required=True, 
                       help="Path to processed directory containing graph files")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for plots")
    parser.add_argument("--fmin", type=float, default=1,
                       help="Minimum frequency (default: 1)")
    parser.add_argument("--fmax", type=float, default=100,
                       help="Maximum frequency (default: 100)")
    
    args = parser.parse_args()
    
    # Setup (similar to plot_graph_analysis)
    if args.output_dir is None:
        output_dir = os.path.join(args.processed_dir, 'comprehensive_analysis')
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Comprehensive MEG Graph Analysis ===")
    print(f"Input directory: {args.processed_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load scout information
    try:
        scout_positions, scout_coords_df, node_labels = load_scout_names()
        print(f"Loaded {len(node_labels)} scout regions")
        
        # Convert positions to dictionary (similar to plot_graph_analysis)
        node_positions_dict = {i: pos for i, pos in enumerate(scout_positions.values())}
        node_labels_dict = {i: label for i, label in enumerate(scout_positions.keys())}
        
    except Exception as e:
        print(f"Warning: Could not load scout information: {e}")
        scout_positions, scout_coords_df, node_labels = None, None, None
        node_positions_dict, node_labels_dict = None, None
    
    # Load all graphs
    graphs_by_file, all_graphs = load_graphs_from_directory(args.processed_dir)
    
    if not all_graphs:
        print("No graph files found!")
        return
    
    # Filter valid graphs (similar to plot_graph_analysis)
    valid_graphs = [g for g in all_graphs if g.y is not None]
    print(f"Loaded {len(all_graphs)} total graphs from {len(graphs_by_file)} files")
    print(f"Valid graphs: {len(valid_graphs)}")
    print(f'Number of graphs ON: {len([g for g in valid_graphs if g.y.item() == 1])}')
    print(f'Number of graphs OFF: {len([g for g in valid_graphs if g.y.item() == 0])}')
    
    # 1. CONNECTIVITY MATRIX ANALYSIS
    print("\n1. Creating connectivity matrix analysis...")
    avg_conn_on, avg_conn_off = plot_connectivity_matrices_comprehensive(
        all_graphs, output_dir, node_labels)
    
    # 2. RAW NODE FEATURES FOR SPECIFIC NODES
    print("\n2. Creating node features plot for S2 L, S2 R, MCC...")
    plot_specific_nodes_raw_features(all_graphs, output_dir, node_labels, args.fmax)
    
    # 3. GRAPH STRUCTURE VISUALIZATION
    print("\n3. Creating graph structure visualization...")
    if scout_positions and node_labels:
        plot_graph_structure_visualization(all_graphs, output_dir, scout_positions, node_labels)

    # Create summary (similar to plot_graph_analysis)
    stim_on_count = sum(1 for g in valid_graphs if g.y.item() == 1)
    stim_off_count = sum(1 for g in valid_graphs if g.y.item() == 0)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Total graphs analyzed: {len(valid_graphs)}")
    print(f"Stimulation ON: {stim_on_count}")
    print(f"Stimulation OFF: {stim_off_count}")
    print(f"Output files saved in: {output_dir}")
    print("\nGenerated plots:")
    print("- connectivity_matrix_stimulation_ON.png")
    print("- connectivity_matrix_stimulation_OFF.png")  
    print("- connectivity_matrix_difference.png")
    print("- selected_node_features.png")
    print("- graph_structure_visualization.png")

if __name__ == "__main__":
    main()
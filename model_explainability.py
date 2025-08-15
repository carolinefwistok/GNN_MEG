import numpy as np
import torch
import time
from torch_geometric.loader import DataLoader
from dataset_wrapper import LoadedGraphsDataset
import argparse
from model import GNN
from explain import *
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
import pandas as pd
from openpyxl import load_workbook, Workbook
import os
import traceback
import glob
from pathlib import Path
import seaborn as sns
import scipy.io
import fcntl
import matplotlib.patheffects as path_effects
from torch_geometric.utils import to_networkx

# MPI imports
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    print("Warning: mpi4py not available. Running in serial mode.")

def load_scout_coordinates():
    """
    Load scout names and coordinates for brain visualization
    """
    try:
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

        return scout_positions, cleaned_descriptions
        
    except Exception as e:
        print(f"Error loading scout coordinates: {e}")
        return None, Nonef

def create_subgraph_visualization(graph, most_common_nodes, most_common_labels, patient_code, stim_type, output_dir, rank, scout_positions=None, node_labels=None):
    """
    Create a single graph visualization with highlighted important nodes using scout coordinates.

    Args:
        graph: PyTorch Geometric graph
        most_common_nodes: List of important node indices
        most_common_labels: List of important node labels
        patient_code: Patient identifier
        stim_type: Stimulation type
        output_dir: Directory to save the plot
        rank: MPI rank
        scout_positions: Dictionary of scout positions
        node_labels: List of node labels
    """
    try:
        # Create output directory for plots
        plots_dir = os.path.join(output_dir, 'subgraph_plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Convert to NetworkX graph
        edge_index = graph.edge_index.cpu().numpy()
        edge_attr = graph.edge_attr.cpu().numpy() if graph.edge_attr is not None else None
        
        G = nx.Graph()
        
        # Add nodes
        num_nodes = graph.x.shape[0]
        for i in range(num_nodes):
            G.add_node(i)
        
        # Add edges with weights
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[:, i]
            weight = edge_attr[i] if edge_attr is not None else 1.0
            G.add_edge(src, dst, weight=weight)
        
        # Setup positions using scout coordinates
        if scout_positions and node_labels:
            # Create position dictionary mapping node indices to coordinates
            pos = {}
            for i in range(min(num_nodes, len(node_labels))):
                scout_name = node_labels[i]
                if scout_name in scout_positions:
                    pos[i] = scout_positions[scout_name]
                else:
                    # Fallback to random position
                    pos[i] = np.random.rand(2)
            
            # Ensure all nodes have positions
            for node in G.nodes():
                if node not in pos:
                    pos[node] = np.random.rand(2)
        else:
            # Use spring layout as fallback
            try:
                pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
            except:
                pos = nx.random_layout(G, seed=42)
        
        # Create single figure
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Prepare edge weights for visualization
        if edge_attr is not None and G.edges():
            edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
            if edge_weights:
                min_weight, max_weight = min(edge_weights), max(edge_weights)
                if max_weight > min_weight:
                    normalized_edge_widths = [(w - min_weight) / (max_weight - min_weight) * 2 + 0.2 for w in edge_weights]
                else:
                    normalized_edge_widths = [0.5] * len(edge_weights)
            else:
                normalized_edge_widths = []
        else:
            normalized_edge_widths = [0.5] * len(G.edges()) if G.edges() else []
        
        # Draw all edges first (in background)
        if G.edges():
            nx.draw_networkx_edges(G, pos, alpha=0.2, width=normalized_edge_widths, 
                                 edge_color='lightgray', ax=ax)
        
        # Draw all nodes (non-important nodes)
        all_nodes = list(G.nodes())
        important_nodes = [node for node in most_common_nodes if node < num_nodes] if most_common_nodes else []
        non_important_nodes = [node for node in all_nodes if node not in important_nodes]
        
        # Draw non-important nodes
        if non_important_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=non_important_nodes, 
                                 node_color='lightblue', node_size=100, 
                                 alpha=0.6, ax=ax, edgecolors='gray', linewidths=0.5)
        
        # Highlight important nodes
        if important_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=important_nodes, 
                                 node_color='red', node_size=300, alpha=0.9, ax=ax,
                                 edgecolors='darkred', linewidths=2)
            
            # Add labels for important nodes only
            if node_labels:
                important_labels = {}
                for node in important_nodes:
                    if node < len(node_labels):
                        important_labels[node] = node_labels[node]
                    else:
                        important_labels[node] = f'N{node}'
                
                # Draw labels with white outline for better visibility
                label_objects = nx.draw_networkx_labels(G, pos, labels=important_labels, 
                                                      font_size=10, font_weight='bold', 
                                                      font_color='black', ax=ax)
                
                # Add white outline to labels
                for label in label_objects.values():
                    label.set_path_effects([
                        path_effects.Stroke(linewidth=3, foreground='white'),
                        path_effects.Normal()
                    ])
        
        # Set title
        ax.set_title(f'{patient_code}_{stim_type} - Brain Network\n'
                    f'({num_nodes} nodes, {G.number_of_edges()} edges, {len(important_nodes)} important nodes)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        ax.axis('off')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=12, label=f'Important Nodes ({len(important_nodes)})', 
                      markeredgecolor='darkred', markeredgewidth=2),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                      markersize=8, label=f'Other Nodes ({len(non_important_nodes)})', 
                      markeredgecolor='gray'),
            plt.Line2D([0], [0], color='lightgray', linewidth=2, label='Connections')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=12, 
                 bbox_to_anchor=(0.02, 0.98))
        
        # Add information text
        info_text = f"Patient: {patient_code}\nStimulation: {stim_type}\n"
        info_text += f"Total nodes: {num_nodes}\nTotal edges: {G.number_of_edges()}\n"
        info_text += f"Important nodes: {len(important_nodes)}"
        if most_common_labels and len(most_common_labels) > 0:
            info_text += f"\nKey regions:\n" + "\n".join([f"• {label}" for label in most_common_labels[:5]])
        
        # Position info box in bottom left
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=11,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor='gray'),
               verticalalignment='bottom')
        
        # Save plot
        plot_filename = f"brain_{patient_code}_{stim_type}_rank{rank}.png"
        plot_path = os.path.join(plots_dir, plot_filename)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[Rank {rank}] Brain plot saved: {plot_path}")
        return plot_path
        
    except Exception as e:
        print(f"[Rank {rank}] Error creating brain visualization for {patient_code}_{stim_type}: {e}")
        traceback.print_exc()
        return None

def group_graphs_by_file(dataset, file_list=None):
    """
    Group graphs by patient_code and stimulation_type (i.e., by original file).
    
    Args:
        dataset: LoadedGraphsDataset
        file_list: Optional list of specific files to process (format: ["PT01_BURST", "PT03_TONIC"])
    
    Returns:
        dict: {(patient_code, stim_type): [graph_indices]}
    """
    file_groups = {}
    
    print(f"Grouping {len(dataset)} graphs by file...")
    
    for idx, graph in enumerate(dataset):
        # Extract patient info
        patient_code = getattr(graph, 'extracted_patient_code', None)
        stim_type = getattr(graph, 'extracted_stim_type', None)
        
        # Fallback methods if direct attributes don't exist
        if not patient_code or not stim_type:
            patient_code, stim_type = extract_patient_info_from_graph(graph, idx, dataset)
        
        if patient_code == 'UNKNOWN' or stim_type == 'UNKNOWN':
            print(f"Warning: Could not extract patient info for graph {idx}")
            continue
        
        file_key = (patient_code, stim_type)
        
        # Filter by file_list if provided
        if file_list is not None:
            file_name = f"{patient_code}_{stim_type}"
            if file_name not in file_list:
                continue
        
        if file_key not in file_groups:
            file_groups[file_key] = []
        file_groups[file_key].append(idx)
    
    print(f"Found {len(file_groups)} unique files:")
    for (patient, stim), indices in file_groups.items():
        print(f"  {patient}_{stim}: {len(indices)} graphs")
    
    return file_groups

def extract_patient_info_from_graph(graph, graph_idx, dataset):
    """
    Extract patient code and stimulation type from a single graph.
    """
    try:
        # Method 1: Use pre-extracted info
        patient_code = getattr(graph, 'extracted_patient_code', None)
        stim_type = getattr(graph, 'extracted_stim_type', None)
        
        if patient_code and stim_type:
            return patient_code, stim_type
        
        # Method 2: Use original filename
        original_filename = getattr(graph, 'original_filename', None)
        if original_filename:
            return parse_filename_direct(original_filename)
        
        # Method 3: Use dataset's graph_files list
        if hasattr(dataset, 'graph_files') and graph_idx < len(dataset.graph_files):
            filename = dataset.graph_files[graph_idx]
            return parse_filename_direct(filename)
        
        # Method 4: Reconstruct from folder
        if hasattr(dataset, 'graphs_folder'):
            pt_files = sorted(glob.glob(os.path.join(dataset.graphs_folder, "*.pt")))
            if graph_idx < len(pt_files):
                filename = pt_files[graph_idx]
                return parse_filename_direct(filename)
        
        return 'UNKNOWN', 'UNKNOWN'
        
    except Exception as e:
        print(f"Warning: Error extracting patient info for graph {graph_idx}: {e}")
        return 'UNKNOWN', 'UNKNOWN'

def parse_filename_direct(filename):
    """
    Parse filename to extract patient code and stimulation type.
    Expected format: graph_PT04_BURST_27.pt or similar
    """
    try:
        if filename is None:
            return "UNKNOWN", "UNKNOWN"
            
        # Remove path if present
        filename = os.path.basename(filename)
        
        # Remove .pt suffix
        if filename.endswith('.pt'):
            filename = filename[:-3]
        
        # Expected format: graph_PT04_BURST_27
        if filename.startswith('graph_'):
            filename = filename[6:]  # Remove 'graph_'
        
        # Split by underscore: PT04_BURST_27
        parts = filename.split('_')
        
        if len(parts) >= 2:
            patient_code = parts[0]  # PT04
            stim_type = parts[1]     # BURST
            return patient_code, stim_type
        else:
            return "UNKNOWN", "UNKNOWN"
            
    except Exception as e:
        print(f"Error parsing filename {filename}: {e}")
        return "UNKNOWN", "UNKNOWN"

def save_single_result_to_excel(result, output_excel_path, n_min):
    """
    Save a single result to Excel file immediately, updating existing data.
    Uses file locking to prevent race conditions in MPI.
    
    Args:
        result: Single result dictionary
        output_excel_path: Path to output Excel file
        n_min: Number of top nodes/labels considered
    """
    try:
        if result is None:
            return
        
        # Prepare single row data
        row_data = {
            'Patient_Code': result['patient_code'],
            'Stimulation_Type': result['stimulation_type'],
            'File_Name': f"{result['patient_code']}_{result['stimulation_type']}",
            'Total_Graphs': result['total_graphs'],
            'Successful_Graphs': result['successful_graphs'],
            'Failed_Graphs': result['total_graphs'] - result['successful_graphs'],
            'Most_Common_Nodes': str(result['most_common_nodes']),
            'Most_Common_Labels': str(result['most_common_labels']),
            'Node_Frequencies': str(result.get('node_frequencies', {})),
            'Label_Frequencies': str(result.get('label_frequencies', {})),
            'Avg_Fidelity_Score': result['avg_fidelity_score'],
            'Avg_Sparsity_Score': result['avg_sparsity_score'],
            'Runtime_Minutes': result['runtime_minutes'],
            'N_Min': n_min,
            'Rank': result['rank'],
            'Plot_Path': result.get('plot_path', ''),
            'Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        new_row_df = pd.DataFrame([row_data])
        
        # Use file locking to prevent race conditions
        lock_file_path = output_excel_path + '.lock'
        
        with open(lock_file_path, 'w') as lock_file:
            try:
                # Acquire exclusive lock
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                
                # Check if Excel file exists and read existing data
                if os.path.exists(output_excel_path):
                    try:
                        # Read from Individual_Files sheet to avoid summary row
                        existing_df = pd.read_excel(output_excel_path, sheet_name='Individual_Files')
                        
                        # Remove existing entry for this file if it exists
                        file_name = row_data['File_Name']
                        existing_df_filtered = existing_df[existing_df['File_Name'] != file_name]
                        
                        # Add new row
                        updated_df = pd.concat([existing_df_filtered, new_row_df], ignore_index=True)
                        print(f"[Rank {result['rank']}] Updated existing file {file_name}")
                        
                    except Exception as e:
                        print(f"Error reading existing Excel: {e}")
                        updated_df = new_row_df
                        print(f"[Rank {result['rank']}] Created new Excel with {file_name}")
                else:
                    updated_df = new_row_df
                    print(f"[Rank {result['rank']}] Created new Excel file")
                
                # Sort by File_Name
                updated_df = updated_df.sort_values('File_Name')
                
                # Calculate updated summary statistics
                summary_stats = {
                    'Total_Files': len(updated_df),
                    'Total_Graphs': updated_df['Total_Graphs'].sum(),
                    'Total_Successful_Graphs': updated_df['Successful_Graphs'].sum(),
                    'Total_Failed_Graphs': updated_df['Failed_Graphs'].sum(),
                    'Overall_Success_Rate': updated_df['Successful_Graphs'].sum() / updated_df['Total_Graphs'].sum() if updated_df['Total_Graphs'].sum() > 0 else 0,
                    'Avg_Fidelity_Score': updated_df['Avg_Fidelity_Score'].mean(),
                    'Std_Fidelity_Score': updated_df['Avg_Fidelity_Score'].std(),
                    'Avg_Sparsity_Score': updated_df['Avg_Sparsity_Score'].mean(),
                    'Std_Sparsity_Score': updated_df['Avg_Sparsity_Score'].std(),
                    'Total_Runtime_Hours': updated_df['Runtime_Minutes'].sum() / 60,
                    'Analysis_Date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Create summary row
                summary_row = {
                    'Patient_Code': 'SUMMARY',
                    'Stimulation_Type': 'ALL',
                    'File_Name': 'TOTAL_SUMMARY',
                    'Total_Graphs': summary_stats['Total_Graphs'],
                    'Successful_Graphs': summary_stats['Total_Successful_Graphs'],
                    'Failed_Graphs': summary_stats['Total_Failed_Graphs'],
                    'Most_Common_Nodes': 'See Label_Frequencies sheet',
                    'Most_Common_Labels': 'See Label_Frequencies sheet',
                    'Node_Frequencies': 'See Label_Frequencies sheet',
                    'Label_Frequencies': 'See Label_Frequencies sheet',
                    'Avg_Fidelity_Score': summary_stats['Avg_Fidelity_Score'],
                    'Avg_Sparsity_Score': summary_stats['Avg_Sparsity_Score'],
                    'Runtime_Minutes': summary_stats['Total_Runtime_Hours'] * 60,
                    'N_Min': n_min,
                    'Rank': 'SUMMARY',
                    'Plot_Path': 'N/A',
                    'Timestamp': summary_stats['Analysis_Date']
                }
                
                # Create results with summary
                results_with_summary = pd.concat([updated_df, pd.DataFrame([summary_row])], ignore_index=True)
                
                # Calculate label frequencies across all current results
                all_labels = []
                for _, row in updated_df.iterrows():
                    if result.get('all_labels'):
                        all_labels.extend(result['all_labels'])
                
                if all_labels:
                    label_counter = Counter(all_labels)
                    label_freq_data = []
                    for label, count in label_counter.most_common():
                        label_freq_data.append({
                            'Brain_Region': label,
                            'Frequency': count,
                            'Percentage': (count / len(all_labels)) * 100 if all_labels else 0
                        })
                    label_freq_df = pd.DataFrame(label_freq_data)
                else:
                    label_freq_df = pd.DataFrame(columns=['Brain_Region', 'Frequency', 'Percentage'])
                
                # Create summary statistics dataframe
                summary_df = pd.DataFrame([summary_stats])
                
                # Save to Excel with multiple sheets
                with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
                    # Main results sheet
                    results_with_summary.to_excel(writer, sheet_name='Results', index=False)
                    
                    # Summary statistics sheet
                    summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
                    
                    # Label frequencies sheet
                    label_freq_df.to_excel(writer, sheet_name='Label_Frequencies', index=False)
                    
                    # Individual file details sheet (without summary row)
                    updated_df.to_excel(writer, sheet_name='Individual_Files', index=False)
                
                print(f"[Rank {result['rank']}] ✅ Excel updated with {file_name}")
                
            finally:
                # Release lock
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        
        # Clean up lock file
        try:
            os.remove(lock_file_path)
        except:
            pass
            
    except Exception as e:
        print(f"Error saving single result to Excel: {e}")
        traceback.print_exc()

def process_file_graphs(file_key, graph_indices, dataset, model, best_result, input_type, n_min, rank, output_dir, scout_positions=None, node_labels=None):
    """
    Process all graphs belonging to a single file and return averaged results.
    MODIFIED: Now saves to Excel immediately after processing each file and creates enhanced plots.
    
    Args:
        file_key: (patient_code, stim_type)
        graph_indices: List of graph indices belonging to this file
        dataset: LoadedGraphsDataset
        model: Trained GNN model
        best_result: Training result object
        input_type: Type of input data
        n_min: Minimum number of important nodes
        rank: MPI rank
        output_dir: Output directory for plots
        scout_positions: Dictionary of scout positions
        node_labels: List of node labels
    
    Returns:
        dict: Averaged results for the file
    """
    try:
        start_time = time.time()
        patient_code, stim_type = file_key
        
        print(f"[Rank {rank}] Processing file {patient_code}_{stim_type} with {len(graph_indices)} graphs")
        
        # Initialize accumulators
        all_fidelity_scores = []
        all_sparsity_scores = []
        all_most_common_nodes = []
        all_most_common_labels = []
        successful_graphs = 0
        device = next(model.parameters()).device
        
        # Process each graph in the file
        for i, graph_idx in enumerate(graph_indices):
            try:
                graph = dataset[graph_idx].to(device)
                label = graph.y.item()
                
                # Perform explainability analysis on this graph
                avg_accuracy_normal, avg_accuracy_occluded, most_common_nodes, most_common_labels, accuracy_difference, node_frequencies, avg_fidelity_ratio, avg_fidelity_score = calculate_explain_and_accuracies(
                    graph, model, dataset, best_result, input_type, n_min
                )
                
                # Calculate sparsity
                sparsity_score = sparsity(graph, most_common_nodes)
                
                # Accumulate results
                if avg_fidelity_score is not None:
                    all_fidelity_scores.append(avg_fidelity_score)
                if sparsity_score is not None:
                    all_sparsity_scores.append(sparsity_score)
                if most_common_nodes:
                    all_most_common_nodes.extend(most_common_nodes)
                if most_common_labels:
                    all_most_common_labels.extend(most_common_labels)
                
                successful_graphs += 1
                
                # Progress reporting
                if (i + 1) % 10 == 0 or (i + 1) == len(graph_indices):
                    print(f"[Rank {rank}] {patient_code}_{stim_type}: Processed {i + 1}/{len(graph_indices)} graphs")
                
            except Exception as e:
                print(f"[Rank {rank}] Error processing graph {graph_idx} in file {patient_code}_{stim_type}: {e}")
                continue
        
        # Calculate averages and most common elements
        avg_fidelity_score = np.mean(all_fidelity_scores) if all_fidelity_scores else None
        avg_sparsity_score = np.mean(all_sparsity_scores) if all_sparsity_scores else None
        
        # Get most common nodes and labels across all graphs in this file
        node_counter = Counter(all_most_common_nodes)
        label_counter = Counter(all_most_common_labels)
        
        # Get top n_min most common
        most_common_nodes_file = [node for node, _ in node_counter.most_common(n_min)]
        most_common_labels_file = [label for label, _ in label_counter.most_common(n_min)]
        
        # Create enhanced visualization plot for this file
        plot_path = None
        if successful_graphs > 0 and most_common_nodes_file:
            try:
                representative_graph = dataset[graph_indices[0]].to(device)
                plot_path = create_subgraph_visualization(
                    representative_graph, most_common_nodes_file, most_common_labels_file,
                    patient_code, stim_type, output_dir, rank, scout_positions, node_labels
                )
            except Exception as e:
                print(f"[Rank {rank}] Error creating enhanced visualization for {patient_code}_{stim_type}: {e}")
        
        # Calculate runtime
        end_time = time.time()
        runtime_minutes = (end_time - start_time) / 60
        
        # Create result dictionary
        result = {
            'patient_code': patient_code,
            'stimulation_type': stim_type,
            'total_graphs': len(graph_indices),
            'successful_graphs': successful_graphs,
            'most_common_nodes': most_common_nodes_file,
            'most_common_labels': most_common_labels_file,
            'node_frequencies': dict(node_counter.most_common(n_min)),
            'label_frequencies': dict(label_counter.most_common(n_min)),
            'all_labels': list(label_counter.keys()),  # All unique labels for summary
            'avg_fidelity_score': avg_fidelity_score,
            'avg_sparsity_score': avg_sparsity_score,
            'runtime_minutes': runtime_minutes,
            'rank': rank,
            'plot_path': plot_path
        }
        
        print(f"[Rank {rank}] Completed file {patient_code}_{stim_type}:")
        print(f"  Successful graphs: {successful_graphs}/{len(graph_indices)}")
        print(f"  Avg fidelity: {avg_fidelity_score:.4f}" if avg_fidelity_score else "  Avg fidelity: None")
        print(f"  Avg sparsity: {avg_sparsity_score:.4f}" if avg_sparsity_score else "  Avg sparsity: None")
        print(f"  Runtime: {runtime_minutes:.2f} minutes")
        print(f"  Top nodes: {most_common_nodes_file}")
        print(f"  Top labels: {most_common_labels_file}")
        if plot_path:
            print(f"  Enhanced plot saved: {plot_path}")
        
        return result
        
    except Exception as e:
        print(f"[Rank {rank}] Error processing file {patient_code}_{stim_type}: {e}")
        traceback.print_exc()
        return None

def explain_by_file_mpi(model_path, best_result_path, dataset, input_type, output_excel_path, n_min=5, file_list=None):
    """
    MPI-parallelized explainability analysis by file.
    MODIFIED: Now saves results immediately after each file is processed with enhanced visualizations.
    
    Args:
        model_path: Path to saved model
        best_result_path: Path to best training result
        dataset: LoadedGraphsDataset
        input_type: Type of input data
        output_excel_path: Path to output Excel file
        n_min: Number of top nodes/labels to consider
        file_list: Optional list of specific files to process
    """
    # Setup MPI
    if MPI_AVAILABLE:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        comm, rank, size = None, 0, 1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Rank {rank}] Using device: {device}")
    
    # Create output directory for plots
    output_dir = os.path.dirname(output_excel_path)
    
    # Load scout coordinates (only rank 0 needs to do this initially)
    if rank == 0:
        scout_positions, node_labels = load_scout_coordinates()
        if scout_positions:
            print(f"Loaded {len(scout_positions)} scout positions")
        else:
            print("Warning: Could not load scout coordinates - using fallback layout")
    else:
        scout_positions, node_labels = None, None
    
    # Broadcast scout information to all ranks
    if MPI_AVAILABLE:
        scout_positions = comm.bcast(scout_positions, root=0)
        node_labels = comm.bcast(node_labels, root=0)
    
    # Load model and best result
    checkpoint = torch.load(best_result_path, weights_only=False)
    best_result = checkpoint['best_result']
    
    if rank == 0:
        print(f"Loaded best result with config keys: {list(best_result.config.keys())}")
    
    # Initialize model
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
    
    # Load model weights
    with best_result.checkpoint.as_directory() as checkpoint_dir:
        state_dict = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pt'), weights_only=True)
        model.load_state_dict(state_dict['model_state'])
        model.to(device)
    
    model.eval()
    print(f"[Rank {rank}] Model loaded and set to evaluation mode")
    
    # Group graphs by file (only rank 0 does this)
    if rank == 0:
        file_groups = group_graphs_by_file(dataset, file_list)
        file_keys = list(file_groups.keys())
        
        if not file_groups:
            print("No files found to process!")
            if MPI_AVAILABLE:
                # Send empty list to all ranks
                for r in range(1, size):
                    comm.send([], dest=r, tag=0)
            return
        
        print(f"Found {len(file_keys)} files to process with {size} ranks")
    else:
        file_keys = None
        file_groups = None
    
    # Broadcast file information to all ranks
    if MPI_AVAILABLE:
        if rank == 0:
            # Send file assignments to each rank
            files_per_rank = []
            for r in range(size):
                start_idx = r * len(file_keys) // size
                end_idx = (r + 1) * len(file_keys) // size
                if r == size - 1:  # Last rank gets any remainder
                    end_idx = len(file_keys)
                
                rank_file_keys = file_keys[start_idx:end_idx]
                rank_file_groups = {k: file_groups[k] for k in rank_file_keys}
                files_per_rank.append(rank_file_groups)
            
            # Send to other ranks
            for r in range(1, size):
                comm.send(files_per_rank[r], dest=r, tag=0)
            
            # Keep rank 0's assignment
            my_file_groups = files_per_rank[0]
        else:
            # Receive assignment from rank 0
            my_file_groups = comm.recv(source=0, tag=0)
    else:
        my_file_groups = file_groups
    
    if not my_file_groups:
        print(f"[Rank {rank}] No files assigned to this rank")
    else:
        print(f"[Rank {rank}] Processing {len(my_file_groups)} files: {list(my_file_groups.keys())}")
        
        # Process assigned files and save immediately
        for i, (file_key, graph_indices) in enumerate(my_file_groups.items()):
            result = process_file_graphs(
                file_key, graph_indices, dataset, model, best_result, 
                input_type, n_min, rank, output_dir, scout_positions, node_labels
            )
            
            # SAVE IMMEDIATELY after processing each file
            if result is not None:
                save_single_result_to_excel(result, output_excel_path, n_min)
                print(f"[Rank {rank}] ✅ Results for {file_key[0]}_{file_key[1]} saved to Excel")
            else:
                print(f"[Rank {rank}] ❌ Failed to process {file_key[0]}_{file_key[1]}")
            
            print(f"[Rank {rank}] Completed file {i+1}/{len(my_file_groups)}")
    
    # Synchronize all ranks before finishing
    if MPI_AVAILABLE:
        comm.barrier()
    
    if rank == 0:
        print("✅ All files processed and saved to Excel with enhanced visualizations!")
        print(f"📁 Excel file location: {output_excel_path}")
        print(f"🎨 Enhanced plots saved in: {os.path.join(output_dir, 'subgraph_plots')}")

def main():
    """
    Main function for file-based explainability analysis with enhanced visualization.
    """
    parser = argparse.ArgumentParser(description="File-based explainability analysis with enhanced brain visualization")
    parser.add_argument("--processed_dir", required=True, help="Directory containing processed graphs")
    parser.add_argument("--model_path", required=True, help="Path to saved model")
    parser.add_argument("--best_result_path", required=True, help="Path to best training result")
    parser.add_argument("--output_excel", required=True, help="Path to output Excel file")
    parser.add_argument("--input_type", default="scout", help="Type of input data")
    parser.add_argument("--n_min", type=int, default=5, help="Number of top nodes/labels to consider")
    parser.add_argument("--files", nargs="*", help="Specific files to process (e.g., PT01_BURST PT03_TONIC)")
    
    args = parser.parse_args()
    
    # Setup MPI for initial prints
    if MPI_AVAILABLE:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        rank = 0
        size = 1
    
    if rank == 0:
        print(f"Starting file-based explainability analysis with {size} process(es)")
        print(f"Processed directory: {args.processed_dir}")
        print(f"Model path: {args.model_path}")
        print(f"Best result path: {args.best_result_path}")
        print(f"Output Excel: {args.output_excel}")
        print(f"Input type: {args.input_type}")
        print(f"N_min: {args.n_min}")
        if args.files:
            print(f"Specific files to process: {args.files}")
        
        # Create output directory
        os.makedirs(os.path.dirname(args.output_excel), exist_ok=True)
    
    # Load dataset
    dataset = LoadedGraphsDataset(args.processed_dir)
    
    if rank == 0:
        print(f"Loaded dataset with {len(dataset)} graphs")
    
    if len(dataset) == 0:
        if rank == 0:
            print("No graphs found in dataset!")
        return
    
    # Run analysis
    explain_by_file_mpi(
        model_path=args.model_path,
        best_result_path=args.best_result_path,
        dataset=dataset,
        input_type=args.input_type,
        output_excel_path=args.output_excel,
        n_min=args.n_min,
        file_list=args.files
    )
    
    if rank == 0:
        print("Enhanced file-based explainability analysis completed!")

if __name__ == "__main__":
    main()
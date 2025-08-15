import torch
import os
from torch_geometric.data import Dataset, Data
import re

class LoadedGraphsDataset(Dataset):
    """
    A Dataset-like wrapper for already-saved graphs in a folder.
    Mimics the interface of MEGGraphs for downstream use.
    """
    def __init__(self, graphs_folder, add_patient_ids=False, file_to_patient_mapping=None):
        """
        Args:
            graphs_folder (str): Path to the folder containing .pt graph files.
            add_patient_ids (bool): Whether to add patient IDs to graphs that don't have them.
            file_to_patient_mapping (dict): Mapping from file index to patient info.
        """
        
        self.graphs_folder = graphs_folder
        self.add_patient_ids = add_patient_ids
        self.file_to_patient_mapping = file_to_patient_mapping or self._get_default_mapping()
        
        self.graph_files = sorted([f for f in os.listdir(graphs_folder) if f.endswith('.pt')])
        
        # Detect naming convention
        self.naming_convention = self._detect_naming_convention()
        print(f"Detected naming convention: {self.naming_convention}")
        
        self.graphs = []
        self._load_graphs()
        
        self.filenames = [getattr(g, 'filename', None) for g in self.graphs]
        self.actual_epochs_per_file = None
        
        print(f"Loaded {len(self.graphs)} valid graphs from {graphs_folder}")

    def _get_default_mapping(self):
        """Default file index to patient mapping based on your provided list"""
        return {
            0: ("PT01", "BURST"),
            1: ("PT03", "BURST"),
            2: ("PT03", "TONIC"),
            3: ("PT04", "BURST"),
            4: ("PT04", "TONIC"),
            5: ("PT05", "BURST"),
            6: ("PT06", "BURST"),
            7: ("PT06", "TONIC"),
            8: ("PT07", "BURST"),
            9: ("PT07", "TONIC"),
            10: ("PT08", "BURST"),
            11: ("PT08", "TONIC"),
            12: ("PT09", "BURST"),
            13: ("PT09", "TONIC"),
            14: ("PT10", "BURST"),
            15: ("PT10", "TONIC"),
            16: ("PT11", "BURST"),
            17: ("PT11", "TONIC"),
            18: ("PTN04", "BURST"),
            19: ("PTN04", "TONIC"),
            20: ("PTN05", "BURST"),
            21: ("PTN05", "TONIC"),
            22: ("PTN06", "BURST"),
            23: ("PTN06", "TONIC"),
            24: ("PTN07", "BURST"),
            25: ("PTN07", "TONIC"),
            26: ("PTN08", "BURST"),
            27: ("PTN09", "BURST"),
            28: ("PTN09", "TONIC"),
            29: ("PTN10", "BURST"),
            30: ("PTN13", "TONIC"),
            31: ("PTN14", "BURST"),
            32: ("PTN14", "TONIC"),
            33: ("PTN15", "BURST"),
            34: ("PTN15", "TONIC")
        }

    def _detect_naming_convention(self):
        """
        Detect whether graphs use old format (graph_{idx_file}_{idx_graph}) 
        or new format (graph_{patient_code}_{stim_type}_{idx_graph})
        """
        if not self.graph_files:
            return "unknown"
        
        # Check first few files to determine pattern
        for filename in self.graph_files[:5]:
            # Remove .pt extension for analysis
            base_name = filename.replace('.pt', '')
            
            # Old format: graph_0_1, graph_15_23, etc.
            old_pattern = r'^graph_(\d+)_(\d+)$'
            
            # New format: graph_PT01_BURST_1, graph_PTN05_TONIC_23, etc.
            new_pattern = r'^graph_(PTN?\d+)_([A-Z]+)_(\d+)$'
            
            if re.match(old_pattern, base_name):
                return "old_format"
            elif re.match(new_pattern, base_name):
                return "new_format"
        
        return "unknown"

    def _extract_info_from_filename(self, filename):
        """
        Extract file index, graph index, and patient info from filename
        Returns: (file_idx, graph_idx, patient_code, stim_type)
        """
        base_name = filename.replace('.pt', '')
        
        if self.naming_convention == "old_format":
            # graph_15_23 -> file_idx=15, graph_idx=23
            match = re.match(r'^graph_(\d+)_(\d+)$', base_name)
            if match:
                file_idx = int(match.group(1))
                graph_idx = int(match.group(2))
                
                # Get patient info from mapping
                if file_idx in self.file_to_patient_mapping:
                    patient_code, stim_type = self.file_to_patient_mapping[file_idx]
                    return file_idx, graph_idx, patient_code, stim_type
                else:
                    print(f"Warning: No patient mapping found for file index {file_idx}")
                    return file_idx, graph_idx, None, None
            
        elif self.naming_convention == "new_format":
            # graph_PT01_BURST_23 -> patient_code=PT01, stim_type=BURST, graph_idx=23
            match = re.match(r'^graph_(PTN?\d+)_([A-Z]+)_(\d+)$', base_name)
            if match:
                patient_code = match.group(1)
                stim_type = match.group(2)
                graph_idx = int(match.group(3))
                
                # Find corresponding file_idx (reverse lookup)
                file_idx = None
                for idx, (p_code, s_type) in self.file_to_patient_mapping.items():
                    if p_code == patient_code and s_type == stim_type:
                        file_idx = idx
                        break
                
                return file_idx, graph_idx, patient_code, stim_type
        
        print(f"Warning: Could not parse filename {filename}")
        return None, None, None, None

    def _load_graphs(self):
        """Load graphs and optionally add patient IDs"""

        total_files = len(self.graph_files)
        valid_graphs_loaded = 0
        invalid_no_features = 0

        for filename in self.graph_files:
            try:
                graph = torch.load(os.path.join(self.graphs_folder, filename), weights_only=False)
                
                if not isinstance(graph, Data):
                    print(f"Warning: {filename} is not a valid Data object and will be skipped.")
                    continue
                
                if getattr(graph, 'x', None) is None:
                    print(f"Warning: {filename} has no node features and will be skipped.")
                    invalid_no_features += 1
                    continue
                
                # Extract information from filename
                file_idx, graph_idx, patient_code, stim_type = self._extract_info_from_filename(filename)
                
                # Check if patient_id is missing and should be added
                needs_patient_id = not hasattr(graph, 'patient_id') or graph.patient_id is None
                
                if self.add_patient_ids and needs_patient_id:
                    if patient_code is not None:
                        # Set patient_id to patient code only (for proper grouping)
                        graph.patient_id = patient_code
                        print(f"Added patient_id '{patient_code}' to graph from {filename}")
                    else:
                        print(f"Warning: Could not determine patient_id for {filename}")
                
                # Set patient_file if missing
                if not hasattr(graph, 'patient_file') or graph.patient_file is None:
                    if file_idx is not None:
                        graph.patient_file = file_idx
                
                # Add filename for reference
                graph.original_filename = filename
                graph.extracted_patient_code = patient_code
                graph.extracted_stim_type = stim_type
                graph.extracted_file_idx = file_idx
                
                self.graphs.append(graph)
                valid_graphs_loaded += 1
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")

        print(f"\n=== GRAPH LOADING SUMMARY ===")
        print(f"Total files processed: {total_files}")
        print(f"Valid graphs loaded: {valid_graphs_loaded}")
        print(f"Invalid graphs (x=None): {invalid_no_features}")

    def get_patient_distribution(self):
        """Get distribution of graphs per patient"""
        patient_counts = {}
        for graph in self.graphs:
            patient_id = getattr(graph, 'patient_id', 'Unknown')
            patient_counts[patient_id] = patient_counts.get(patient_id, 0) + 1
        return patient_counts

    def get_file_distribution(self):
        """Get distribution of graphs per file"""
        file_counts = {}
        for graph in self.graphs:
            file_idx = getattr(graph, 'patient_file', 'Unknown')
            patient_code = getattr(graph, 'extracted_patient_code', 'Unknown')
            stim_type = getattr(graph, 'extracted_stim_type', 'Unknown')
            key = f"{file_idx} ({patient_code} {stim_type})"
            file_counts[key] = file_counts.get(key, 0) + 1
        return file_counts

    def get_stimulation_distribution(self):
        """Get distribution of stimulation vs non-stimulation graphs"""
        stim_counts = {'stim': 0, 'non_stim': 0}
        for graph in self.graphs:
            if hasattr(graph, 'y') and graph.y is not None:
                if graph.y.item() == 1:
                    stim_counts['stim'] += 1
                elif graph.y.item() == 0:
                    stim_counts['non_stim'] += 1
        return stim_counts

    def verify_patient_grouping(self):
        """Verify that patient grouping will work correctly for cross-validation"""
        patient_info = {}
        for graph in self.graphs:
            patient_id = getattr(graph, 'patient_id', None)
            stim_type = getattr(graph, 'extracted_stim_type', None)
            
            if patient_id:
                if patient_id not in patient_info:
                    patient_info[patient_id] = {'stim_types': set(), 'count': 0}
                if stim_type:
                    patient_info[patient_id]['stim_types'].add(stim_type)
                patient_info[patient_id]['count'] += 1
        
        print("\nPatient grouping verification:")
        for patient_id, info in sorted(patient_info.items()):
            stim_types = ', '.join(sorted(info['stim_types']))
            print(f"  {patient_id}: {info['count']} graphs, stim types: {stim_types}")
        
        return patient_info

    @property
    def num_node_features(self):
        for g in self.graphs:
            if g.x is not None:
                return g.x.shape[1]
        return 0

    @property
    def num_classes(self):
        labels = [g.y.item() for g in self.graphs if g.y is not None]
        return len(set(labels))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def get_filenames(self):
        return self.filenames

    def graphs_per_file(self):
        return self.get_file_distribution()

    def stim_graphs_per_file(self):
        file_stim_counts = {}
        for graph in self.graphs:
            if hasattr(graph, 'y') and graph.y is not None and graph.y.item() == 1:
                file_key = f"{getattr(graph, 'patient_file', 'Unknown')}"
                file_stim_counts[file_key] = file_stim_counts.get(file_key, 0) + 1
        return file_stim_counts

    def non_stim_graphs_per_file(self):
        file_non_stim_counts = {}
        for graph in self.graphs:
            if hasattr(graph, 'y') and graph.y is not None and graph.y.item() == 0:
                file_key = f"{getattr(graph, 'patient_file', 'Unknown')}"
                file_non_stim_counts[file_key] = file_non_stim_counts.get(file_key, 0) + 1
        return file_non_stim_counts

    def get_indices_by_label(self, label):
        indices = []
        target_label = 1 if label == 'stim' else 0
        for idx, graph in enumerate(self.graphs):
            if hasattr(graph, 'y') and graph.y is not None and graph.y.item() == target_label:
                indices.append(idx)
        return indices
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def load_roc_data_from_excel(base_dir, frequency_bands):
    """
    Load ROC data from Excel files for different frequency bands.
    
    Args:
        base_dir: Base directory path
        frequency_bands: List of frequency band names
    
    Returns:
        Dictionary with frequency band names as keys and ROC data as values
    """
    roc_data = {}
    
    for band in frequency_bands:
        # Construct the directory path for each frequency band
        model_dir = os.path.join(base_dir, f'gnn_scout_58_MN_grouped_{band}')
        
        # Look for Excel files containing ROC data
        excel_files = []
        if os.path.exists(model_dir):
            excel_files = [f for f in os.listdir(model_dir) 
                          if f.endswith('.xlsx') and ('roc' in f.lower() or 'ROC' in f)]
        
        if not excel_files:
            print(f"Warning: No ROC Excel files found for {band} in {model_dir}")
            continue
        
        # Use the first ROC file found (you can modify this logic if needed)
        excel_file = excel_files[0]
        file_path = os.path.join(model_dir, excel_file)
        
        try:
            # Read the Excel file
            df = pd.read_excel(file_path)
            
            # Check if required columns exist
            required_columns = ['FPR', 'TPR', 'Thresholds']
            if all(col in df.columns for col in required_columns):
                roc_data[band] = {
                    'fpr': df['FPR'].values,
                    'tpr': df['TPR'].values,
                    'thresholds': df['Thresholds'].values
                }
                print(f"Successfully loaded ROC data for {band}: {len(df)} points")
            else:
                print(f"Warning: Required columns not found in {excel_file} for {band}")
                print(f"Available columns: {list(df.columns)}")
                
        except Exception as e:
            print(f"Error loading {excel_file} for {band}: {e}")
    
    return roc_data

def load_roc_data_from_excel_stim_site(base_dir, stim_site_info):
    """
    Load ROC data from Excel files for different stimulation types and recording sites.

    Args:
        base_dir: Base directory path
        stim_site_info: List of stimulation types and site names (TONIC, BURST, Canada, Nijmegen)

    Returns:
        Dictionary with stimulation types and site names as keys and ROC data as values
    """
    roc_data = {}

    for stim_site in stim_site_info:
        # Construct the directory path for each stimulation type and site
        model_dir = os.path.join(base_dir, f'gnn_scout_58_MN_grouped_{stim_site}')

        # Look for Excel files containing ROC data
        excel_files = []
        if os.path.exists(model_dir):
            excel_files = [f for f in os.listdir(model_dir)
                           if f.endswith('.xlsx') and ('roc' in f.lower() or 'ROC' in f)]

        if not excel_files:
            print(f"Warning: No ROC Excel files found for {stim_site} in {model_dir}")
            continue

        # Use the first ROC file found (you can modify this logic if needed)
        excel_file = excel_files[0]
        file_path = os.path.join(model_dir, excel_file)

        try:
            # Read the Excel file
            df = pd.read_excel(file_path)

            # Check if required columns exist
            required_columns = ['FPR', 'TPR', 'Thresholds']
            if all(col in df.columns for col in required_columns):
                roc_data[stim_site] = {
                    'fpr': df['FPR'].values,
                    'tpr': df['TPR'].values,
                    'thresholds': df['Thresholds'].values
                }
                print(f"Successfully loaded ROC data for {stim_site}: {len(df)} points")
            else:
                print(f"Warning: Required columns not found in {excel_file} for {stim_site}")
                print(f"Available columns: {list(df.columns)}")

        except Exception as e:
            print(f"Error loading {excel_file} for {stim_site}: {e}")

    return roc_data

def calculate_auc(fpr, tpr):
    """
    Calculate Area Under the Curve (AUC) using trapezoidal rule.
    """
    return np.trapezoid(tpr, fpr)

def plot_roc_curves(roc_data, output_dir=None):
    """
    Plot ROC curves for all frequency bands.
    
    Args:
        roc_data: Dictionary containing ROC data for each frequency band
        output_dir: Directory to save the plot (optional)
    """
    # Define colors for each frequency band
    colors = {
        'Delta': '#1f77b4',    # Blue
        'Theta': '#ff7f0e',    # Orange
        'Alpha': '#2ca02c',    # Green
        'Beta': '#d62728',     # Red
        'Gamma': '#9467bd',    # Purple
        'Full': '#8c564b',      # Brown
        'Low': '#7f7f7f'        # Gray
    }
    
    # Initialize the plot
    plt.figure(figsize=(10, 8))
    
    # Specify the desired legend order
    desired_order = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma', 'Full', 'Low']
    
    # Plot ROC curves for each frequency band
    auc_values = {}
    plotted_bands = []
    
    for band in desired_order:
        if band in roc_data:
            fpr = roc_data[band]['fpr']
            tpr = roc_data[band]['tpr']
            
            # Calculate AUC
            auc = calculate_auc(fpr, tpr)
            auc_values[band] = auc
            
            # Get color for this band
            color = colors.get(band, None)
            
            # Plot the ROC curve
            plt.plot(fpr, tpr, lw=2, color=color, 
                    label=f'{band}')
            
            plotted_bands.append(band)
            print(f"Plotted {band}")
    
    # Add diagonal reference line (random classifier)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, alpha=0.8, 
             label='Random Classifier')
    
    # Customize the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves for GNN Models Across Different Frequency Bands', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'roc_curves_frequency_bands.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve plot saved to: {output_path}")
    
    # Show the plot
    plt.show()
    
    # Print summary
    print("\n=== ROC Analysis Summary ===")
    print(f"Frequency bands plotted: {len(plotted_bands)}")
    for band in plotted_bands:
        if band in auc_values:
            print(f"{band:>8}: AUC = {auc_values[band]:.2f}")

    # Find best performing model
    if auc_values:
        best_band = max(auc_values.keys(), key=lambda x: auc_values[x])
        print(f"\nBest performing model: {best_band} (AUC = {auc_values[best_band]:.2f})")
    
    return auc_values

def plot_roc_curves_stim_site(roc_data, output_dir=None):
    """
    Plot ROC curves for different stimulation types and recording sites.
    
    Args:
        roc_data: Dictionary containing ROC data for different stimulation types and recording sites
        output_dir: Directory to save the plot (optional)
    """

    # Create a mapping from loaded keys to display labels
    label_mapping = {
        'TONIC': 'Tonic',
        'BURST': 'Burst', 
        'Canada': 'MNI',
        'Nijmegen': 'Donders'
    }

    # Define colors for each frequency label
    colors = {
        'Tonic': "#1b0fbb",    # Blue
        'Burst': "#60C075",    # Orange
        'MNI': "#f740f7",    # Green
        'Donders': "#ac9b08",     # Red
    }
    
    # Initialize the plot
    plt.figure(figsize=(10, 8))
    
    # Specify the desired legend order
    desired_order = ['Tonic', 'Burst', 'MNI', 'Donders']
    
    # Plot ROC curves for each frequency band
    auc_values = {}
    plotted_stim_site_curves = []
    
    # Iterate through loaded data and map to display labels
    for loaded_key, display_label in label_mapping.items():
        if loaded_key in roc_data:  # Check if we have data for this loaded key
            fpr = roc_data[loaded_key]['fpr']
            tpr = roc_data[loaded_key]['tpr']

            # Calculate AUC
            auc = calculate_auc(fpr, tpr)
            auc_values[display_label] = auc
            
            # Get color for this display label
            color = colors.get(display_label, '#000000')
            
            # Plot the ROC curve with display label
            plt.plot(fpr, tpr, lw=3, color=color, 
                    label=f'{display_label}')
            
            plotted_stim_site_curves.append(display_label)
            print(f"Plotted {loaded_key} as {display_label}: AUC = {auc:.2f}")
    
    # Add diagonal reference line (random classifier)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, alpha=0.8, 
             label='Random Classifier')
    
    # Customize the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves for GNN Models Across Stimulation Types and Recording Sites', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'roc_curves_stim_site.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve plot saved to: {output_path}")
    
    # Show the plot
    plt.show()
    
    # Print summary
    print("\n=== ROC Analysis Summary ===")
    print(f"Stimulation types and recording sites plotted: {len(plotted_stim_site_curves)}")
    for stim_site in plotted_stim_site_curves:
        if stim_site in auc_values:
            print(f"{stim_site:>8}: AUC = {auc_values[stim_site]:.2f}")

    # Find best performing model
    if auc_values:
        best_stim_site = max(auc_values.keys(), key=lambda x: auc_values[x])
        print(f"\nBest performing model: {best_stim_site} (AUC = {auc_values[best_stim_site]:.2f})")

    return auc_values

def main():
    """
    Main function to create ROC curves from Excel files.
    """
    # Configuration
    base_dir = '/scratch/cwitstok/Output'
    frequency_bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma', 'Full', 'Low']
    stim_site_analysis = ['TONIC', 'BURST', 'Canada', 'Nijmegen']
    output_dir = '/scratch/cwitstok/Output/ROC_Analysis'
    
    print("Loading ROC data from Excel files...")
    print(f"Base directory: {base_dir}")
    print(f"Frequency bands: {frequency_bands}")
    print(f"Stimulation site analysis: {stim_site_analysis}")

    # Load ROC data from Excel files
    roc_data = load_roc_data_from_excel(base_dir, frequency_bands)
    roc_data_stim_site = load_roc_data_from_excel_stim_site(base_dir, stim_site_analysis)

    if not roc_data:
        print("No ROC data found. Please check the file paths and formats.")
        return
    
    print(f"\nLoaded ROC data for {len(roc_data)} frequency bands")
    print(f"Loaded ROC data for {len(roc_data_stim_site)} stimulation sites")

    # Create ROC curves plot
    auc_values = plot_roc_curves(roc_data, output_dir)
    auc_values_stim_site = plot_roc_curves_stim_site(roc_data_stim_site, output_dir)
    
    # Save AUC summary to CSV
    if auc_values and auc_values_stim_site and output_dir:
        auc_df = pd.DataFrame([
            {'Frequency_Band': band, 'AUC': auc} 
            for band, auc in auc_values.items()
        ])
        auc_summary_path = os.path.join(output_dir, 'auc_summary.csv')
        auc_df.to_csv(auc_summary_path, index=False)
        print(f"AUC summary saved to: {auc_summary_path}")

        auc_stim_site_df = pd.DataFrame([
            {'Stimulation_Site': site, 'AUC': auc}
            for site, auc in auc_values_stim_site.items()
        ])
        auc_stim_site_summary_path = os.path.join(output_dir, 'auc_stim_site_summary.csv')
        auc_stim_site_df.to_csv(auc_stim_site_summary_path, index=False)
        print(f"AUC stimulation site summary saved to: {auc_stim_site_summary_path}")

if __name__ == "__main__":
    main()
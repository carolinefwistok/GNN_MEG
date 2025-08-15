from collections import Counter
import pandas as pd

def rangschik_node_indices_excel(bestandspad, kolomnaam):
    # Excel-bestand inladen
    df = pd.read_excel(bestandspad)

    # Node indices ophalen uit de gewenste kolom en splitsen per komma
    alle_nodes = []
    for cell in df[kolomnaam].dropna():
        alle_nodes.extend(cell.split(','))  # Splits de indices per komma

    # Strip eventuele spaties rond de indices
    alle_nodes = [node.strip() for node in alle_nodes]

    # Tellen van de node-indices
    node_tellingen = Counter(alle_nodes)

    # Rangschikken en weergeven
    ranglijst = node_tellingen.most_common()  # Volledige lijst gerangschikt
    print("Rangschikking van node-indices:")
    for node, count in ranglijst:
        print(f"{node}: {count} keer")

# Voorbeeldgebruik
bestandspad = "explanations_patient6_all_graphs.xlsx"
kolomnaam = "All important Nodes"
rangschik_node_indices_excel(bestandspad, kolomnaam)

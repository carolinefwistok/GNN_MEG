## Classifying Stimulation Status in Chronic Pain Patients Using MEG and Graph Neural Networks
This project focuses on developing a Graph Neural Network (GNN) model to classify stimulation status (ON/OFF) in chronic pain patients with Spinal Cord Stimulation (SCS), using Magnetoencephalography (MEG) data.

The workflow combines MEG preprocessing in Brainstorm, data export and graph dataset generation in Python, and model training/testing on the TU Delft high-performance cluster.

The repository contains scripts for:

* Processing Data – Preprocessing MEG data in Brainstorm and exporting it to Python.
* Graph Datasets – Creating graph datasets with varying input configurations.
* Graph Neural Network – Training and evaluating GNN models on different datasets.
* Explainability – Running SubgraphX-based explainability analyses on trained models.
* Plotting & Visualization – Generating figures to support analysis and interpretation.

For an in-depth explanation of the methods, rationale, and results, please refer to my Master Thesis:
C.F. Witstok, Unravelling Brain Networks in Chronic Pain and Spinal Cord Stimulation through Magnetoencephalography and Graph Neural Networks, TU Delft, 2025. (https://repository.tudelft.nl/record/uuid:28a426ae-582c-47d4-88e1-f96412f09610)

For a detailed technical README describing each script and step, see the README_Feasibility_Study_Caroline_Witstok.pdf included in this repository.

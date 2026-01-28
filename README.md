# Pneumo-Fluor-Toolkit
This branch extracts quantitative single-cell features from the segmentation masks across 3D SIM, 2D WGA+DAPI, and 2D HADA time-series datasets.  
It computes 3D morphology, peptide intensity and patch statistics, probe overlap measures, 2D cell shape metrics, septal vs lateral intensity differences, and nucleoid size/shape/position.  
For time data, it outputs per-cell HADA intensity trajectories and growth profiles.  
All features are exported in CSV-ready tables for downstream probabilistic scoring and visualization in the napari plugin.

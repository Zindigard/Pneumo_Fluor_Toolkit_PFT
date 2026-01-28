# Pneumo-Fluor-Toolkit
This branch implements single-cell segmentation for fluorescence microscopy using Cellpose, StarDist, and Omnipose.  
The pipeline outputs instance masks for each cell and additionally computes per-cell gradient maps (e.g., intensity/edge gradients) aligned to each segmented cell.  
These gradients are extracted per instance (one cell at a time) and saved alongside masks for downstream polarity/edge/texture analysis.  
Results are designed to feed directly into the feature-extraction module.
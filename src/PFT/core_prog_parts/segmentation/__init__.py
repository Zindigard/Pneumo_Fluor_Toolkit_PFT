"""PFT segmentation preparation, annotation, prediction, training, and tuning.

The validated downstream instance-segmentation workflow is implemented by
``segmentation_input_core`` and ``instance_segmentation_core``. It keeps every
dataset/source combination independent and consumes prepared float32 inputs
without model-side percentile normalization.
"""

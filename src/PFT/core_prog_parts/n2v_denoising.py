import numpy as np
from pathlib import Path
from n2v.models import N2V

def denoising_N2V(img_stack, model_dir, model_name):
    """
    N2V donising script.
    Works for: 2D single channel (Y,X), 2D multi (Y,X,C), or Stacks (N,Y,X,C)
    """
    model = N2V(config=None, name=model_name, basedir=str(model_dir))
    
    if img_stack.ndim == 4:
        axes = 'SYXC'
    elif img_stack.ndim == 3:
        axes = 'YXC'
    else:
        # (Y, X)- (Y, X, 1)
        img_stack = img_stack[..., np.newaxis]
        axes = 'YXC'
        
    return model.predict(img_stack, axes=axes)
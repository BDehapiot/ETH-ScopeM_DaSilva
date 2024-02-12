#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage.transform import downscale_local_mean

#%% Functions (GPU) -----------------------------------------------------------

def limit_vram(vram):

    import tensorflow as tf    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    memory_config = tf.config.experimental\
        .VirtualDeviceConfiguration(memory_limit=vram)
    
    if gpus:
        try:
            tf.config.experimental\
                .set_virtual_device_configuration(gpus[0], [memory_config])
        except RuntimeError as e:
            print(e)

#%% Functions -----------------------------------------------------------------

def preprocess(img, msk=None, downscale_factor=4):
    
    # Downscale image
    img = downscale_local_mean(img, downscale_factor)
    
    # Normalize image
    pMax = np.percentile(img, 99.9)
    img[img > pMax] = pMax
    img = (img / pMax).astype(float)
    
    # Downscale mask
    if msk is not None:
        msk = (downscale_local_mean(msk, downscale_factor) >= 1).astype("float")
    
        return img, msk
    
    return img

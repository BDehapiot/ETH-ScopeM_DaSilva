#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io
from pathlib import Path
import segmentation_models as sm
from functions import preprocess
from skimage.transform import rescale

#%% Inputs --------------------------------------------------------------------

# Paths
data_path  = Path(Path.cwd(), "data")
model_path = Path(Path.cwd(), "model_weights.h5")
paths = [
    path for path in data_path.glob("*.tif") 
    if 'prediction' not in path.name
    ]

# Parameters
downscale_factor = 4
thresh = 0.5

# GPU
vram = 2048 # Limit vram (None to deactivate)
if vram:
    from functions import limit_vram
    limit_vram(vram)

#%% Model ---------------------------------------------------------------------

# Define & compile model
model = sm.Unet(
    'resnet34', 
    input_shape=(None, None, 1), 
    classes=1, 
    activation='sigmoid', 
    encoder_weights=None,
    )
model.compile(
    optimizer='adam',
    loss='binary_crossentropy', 
    metrics=['mse']
    )

# Load weights
model.load_weights(model_path) 

#%% Predict -------------------------------------------------------------------

for path in paths:
    
    # Open and preprocess image
    img = io.imread(path)
    img = preprocess(img, downscale_factor=downscale_factor)
    
    # Make prediction and rescale
    prd = model.predict(img[np.newaxis,...]).squeeze()
    prd = rescale(prd, downscale_factor)
    prd = prd > thresh
    
    # Save prediction mask
    io.imsave(
        str(path).replace(".tif", "_prediction.tif"),
        prd.astype("uint8") * 255, check_contrast=False
        )    

#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import segmentation_models as sm
from joblib import Parallel, delayed 

#%% Inputs --------------------------------------------------------------------

# Paths
test_path = Path(Path.cwd(), "data", "test")
model_path = Path(Path.cwd(), "model_weights.h5")

#%% Pre-processing ------------------------------------------------------------

# Open test data
images, masks = [], []
for path in test_path.iterdir():
    if 'mask' in path.name:
        
        # Open masks
        masks.append(io.imread(path).astype("float"))
        
        # Open & normalize images
        image = io.imread(str(path).replace('_mask', ''))
        pMax = np.percentile(image, 99.9)
        image[image > pMax] = pMax
        image = (image / pMax).astype(float)
        images.append(image)                  
        
images = np.stack(images)
masks = np.stack(masks)

#%% Predict -------------------------------------------------------------------

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

# Predict
probs = model.predict(images).squeeze()

#%% Outputs -------------------------------------------------------------------

# # Display 
# viewer = napari.Viewer()
# viewer.add_image(images)
# viewer.add_image(masks)   
# viewer.add_image(probs)

# # Save
# io.imsave(
#     Path(test_path, "prediction.tif"),
#     probs[0].astype("float32"),
#     check_contrast=False,
#     )
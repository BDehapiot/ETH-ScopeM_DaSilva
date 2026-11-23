#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io
from pathlib import Path
from functions import preprocess

#%% Inputs --------------------------------------------------------------------

data_path = Path.cwd() / "data"

#%% Execute -------------------------------------------------------------------

# Paths
prd_paths = list(data_path.glob("*_prediction.tif"))

# Load
imgs, prds = [], []
for prd_path in prd_paths:
    img_path = str(prd_path).replace("_prediction.tif", ".tif")
    img = io.imread(img_path)
    prd = io.imread(prd_path)
    img, prd = preprocess(img, msk=prd, downscale_factor=4)
    imgs.append(img)
    prds.append(prd)
    
# Montage
imgs = np.vstack([np.hstack(imgs[i * 5:(i + 1) * 5]) for i in range(4)])
prds = np.vstack([np.hstack(prds[i * 5:(i + 1) * 5]) for i in range(4)])
mntg = np.hstack((imgs, prds))
    
# Save
io.imsave(
    Path.cwd() / "utils" / "images.jpg",
    (imgs * 255).astype("uint8"), check_contrast=False
    )
io.imsave(
    Path.cwd() / "utils" / "predictions.jpg",
    (prds * 255).astype("uint8"), check_contrast=False
    )
io.imsave(
    Path.cwd() / "utils" / "montage.jpg",
    (mntg * 255).astype("uint8"), check_contrast=False
    )


# # Display
# import napari
# vwr = napari.Viewer()
# vwr.add_image(np.stack(imgs))
# vwr.add_image(np.stack(prds))
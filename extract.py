#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io
from pathlib import Path
from skimage.transform import downscale_local_mean

#%% Inputs --------------------------------------------------------------------

# Paths
local_path = Path("D:\local_DaSilva\data")
trn_img_path = Path(Path.cwd(), "data", "train_images")
trn_msk_path = Path(Path.cwd(), "data", "train_masks")
tst_img_path = Path(Path.cwd(), "data", "test_images")
tst_msk_path = Path(Path.cwd(), "data", "test_masks")
msk_paths = list(local_path.glob("*_mask.tif"))
nMsk = len(msk_paths)

# Parameters
downscale_factor = 4
tst_split = 0.33
np.random.seed(42)

#%% Extract -------------------------------------------------------------------

'''
- Formated for ZeroCostDL4Mic
'''

# Train and test idx
idx = np.random.randint(0, nMsk, nMsk)
nTrn = nMsk - int(np.floor(nMsk * (tst_split)))
trnIdx, tstIdx = idx[0:nTrn], idx[nTrn:]

# Read and format data
for i, path in enumerate(msk_paths):
        
    # Open data
    msk = io.imread(path)
    img = io.imread(str(path).replace("_mask", ""))
        
    # Downscale mask and image
    msk = (downscale_local_mean(msk, downscale_factor) >= 1).astype("float")
    img = downscale_local_mean(img, downscale_factor)
    
    # Normalize image
    pMax = np.percentile(img, 99.9)
    img[img > pMax] = pMax
    img = (img / pMax).astype(float)
    
    # Save data
    name = path.name.replace("_mask", "")
    if (trnIdx == i).any():
        io.imsave(Path(trn_img_path, name),
            img.astype("float32"), check_contrast=False
            )    
        io.imsave(Path(trn_msk_path, name),
            msk.astype("float32"), check_contrast=False
            )        
    if (tstIdx == i).any():
        io.imsave(Path(tst_img_path, name),
            img.astype("float32"), check_contrast=False
            )    
        io.imsave(Path(tst_msk_path, name),
            msk.astype("float32"), check_contrast=False
            )          
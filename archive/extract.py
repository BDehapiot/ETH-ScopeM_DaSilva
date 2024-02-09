#%% Imports -------------------------------------------------------------------

from skimage import io
from pathlib import Path
from skimage.transform import downscale_local_mean

#%% Inputs --------------------------------------------------------------------

# Parameters
rsize_factor = 8

# Paths
data_path  = Path(Path.cwd(), "data")
train_path = Path(Path.cwd(), "data", "train")

#%% Process -------------------------------------------------------------------

for path in data_path.iterdir():
    if path.suffix == '.tif':
        img = io.imread(path)
        rsize = downscale_local_mean(img, rsize_factor)
        io.imsave(
            Path(train_path, path.stem + "_rsize.tif"),
            rsize.astype("uint16"), check_contrast=False,
            )
## Method

<p align="left">
  <img src="utils/montage.jpg" alt="images" width="768" />
</p>

Binary masks of spheroids were generated from 2D brightfield images using a 
custom deep learning pipeline written in Python (v3.10). The model uses a U-Net
 architecture with a ResNet18 encoder backbone. It was implemented in 
 TensorFlow/Keras using the Segmentation Models library 
 (https://github.com/qubvel/segmentation_models). It was trained on 115 images 
 that were annotated manually in a custom Napari interface. Only the overall 
 outline of each spheroid was needed, so the images were downscaled before 
 training and prediction to reduce computation time. The predicted masks were 
 then upscaled back to the original image resolution. All code and training 
 data are available at https://github.com/BDehapiot/ETH-ScopeM_DaSilva.
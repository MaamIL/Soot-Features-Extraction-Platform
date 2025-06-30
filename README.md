# Soot-Profiles-Extraction-Plat
A general platform for creating datasets and running various deep learning models on it for research of extracting soot profiles out of a flame image. Built for Dr. Victor Chernov, Department of Mechanical Engineering, Braude College.
As I'm working for a researcher, changes are done often. The generelization idea is to switch between models and input/output properties easily and fast.

More- TBD


# Soot Profiles Extraction Platform

**Mosheat Alon-Moses, June 2025**

---

## Contents
1. [Introduction](#introduction)
2. [Soot Profiles Extraction Platform](#soot-profiles-extraction-platform)
    - [Architecture](#architecture)
3. [Extracting Soot Features’ Profiles from Flame Images using CNN Deep Learning Neural Network](#extracting-soot-features-profiles-from-flame-images-using-cnn-deep-learning-neural-network)
   - [Introduction](#introduction-1)
   - [Architecture](#architecture-1)
   - [Configuration](#configuration)
   - [Dataset Creation and Data Manipulations](#dataset-creation-and-data-manipulations)
   - [Train Results](#train-results)
   - [Inference and Single Tests](#inference-and-single-tests)

---

## Introduction

The platform was created as part of research extracting soot profiles out of a flame image. Built for Dr. Victor Chernov, Department of Mechanical Engineering, Braude College.

As I'm working for a researcher, changes are often. The generalization idea comes to help me work flexibly, quickly, and efficiently. To switch between models, configurations and training/single-test/inference modes quite easily.

At the beginning, I had only a few samples, so I headed towards pretrained models.

Eventually, we’ve created a synthetic data generator, also uploaded to my GitHub:  
**[MaamIL/SootProfilesGenerateSyntheticData](https://github.com/MaamIL/SootProfilesGenerateSyntheticData)** – Generating synthetic data for deep models that extract soot properties from flame images. A collaboration with Dr. Victor Chernov, Department of Mechanical Engineering, Braude College, and I’ve extracted enough samples to train my own deep learning model.

First, I’ll present platform’s basic architecture and concepts.
Following, I will focus on the CNN encoder-decoder DL (Deep Learning) model and its results.


---

## Soot Profiles Extraction Platform

The main concept of building this platform was to create a flexible platform where I can easily move between different NN models on the same dataset. This need came up as I didn’t have a lot of data, and aimed on pretrained models. I wanted a way of holding all the code and results in the same place for easy maintenance and easy.

For this, I’ve created a folder `Mymodels` holding a class for every model.  
In my main code, I configure the model I want to run (config.model_name) which is identical to the relative class name by using import_module from importlib::

```python
model_module = import_module(f"Mymodels.{config.model_name}")
```

I also have a config class holding all data and model configurations in one place for easy adaptations as needed.

All the code of creating the dataset, logging, running and summarizing the train/single test/inference is wrapping this code and is identical for every run.

The dataset consists of:
- **Input**: RGB image of a flame (laminar axisymmetric flame extracted from a MATLAB `.mat` file)
- **Outputs**: Two matrices representing soot properties per pixel:
  - **Fv** – Soot volume fractions
  - **T** – Temperature

<div align="center">
  <img src="pics for documentation\flames1.png" alt="flame" width = 400>
</div>

More info regarding the data and dataset creation as well as the outputs and summaries can be found in chapters below, under the chosen CNN DL network.

---

## Architecture

<div align="center">
  <img src="pics for documentation\platArchitec.png" alt="Code architechture">
</div>

---

## Extracting Soot Features’ Profiles from Flame Images using CNN Deep Learning Neural Network

### Introduction

This work presents a deep learning approach for pixel-wise regression of soot-related properties directly from RGB flame images. 

Specifically, I designed a CNN-based encoder-decoder architecture that maps a flame RGB image to two heatmaps representing physical fields - soot volume fraction and temperature.

#### Key Contributions
- A deep encoder-decoder architecture for high-resolution per-pixel prediction of physical quantities from flame imagery.
- A supervised learning pipeline based on synthetic data, providing maps of Fv and T.
- A quantitative and qualitative evaluation of model predictions with visualizations.

This framework demonstrates the feasibility of real-time, non-intrusive estimation of combustion properties from visual data, offering potential applications in combustion diagnostics, process monitoring, and intelligent control systems

<div align="center">
  <img src="pics for documentation\cnnflow.png" alt="CNN Flow" width=600>
</div>

---

### Architecture

<div align="center">
  <img src="pics for documentation\cnnblocks.png" alt="CNN Blocks">
</div>

The CNN Encoder-Decoder model includes:
- **Encoder**: 3 initial convolution layers + 4 residual blocks (with skip connections) + 1 additional block
- **Decoder**: 5 residual blocks
- **Final**: 2D convolution followed by a Sigmoid activation function

After preprocessing and building the dataset, training is run.

Training Procedure:
1. Each epoch runs train-validation sets.
2. If best validation loss → save model.
3. If epoch patience is exceeded → stop training.
4. At the end of training: load best model → run test set.

<div align="center">
  <img src="pics for documentation\cnn.jpg" alt="CNN network">
</div>

---

### Configuration

**Loss Function**
<div align="left">
  <img src="pics for documentation\mse.png" alt="mse" width=300>
</div>

The loss is calculated for T and Fv separately, then combined in weight of 50% each:

```python
loss_fv = F.mse_loss(outputs[:, 0, :, :], gts[:, 0, :, :])
loss_T = F.mse_loss(outputs[:, 1, :, :], gts[:, 1, :, :])
loss = 0.5 * loss_fv + 0.5 * loss_T
```

**Scheduler Patience**
```python
torch.optim.lr_scheduler.ReduceLROnPlateau(
    self.config.optimizer, mode='min', factor=0.3, patience=3
)
```
Reduce lr (Learning Rate) in case of plateau in validation loss (no improvements after patience period (3 epochs), then learning rate will be reduced: new_lr = lr * factor (lr*=0.3)

**Epoch Patience**: 
15 epochs without improvement → stop training and continue to the test and plots.

#### Configurations

```python
~~~~~Params for dataset creation~~~~~ (Extracted from all matrices in training data)
global_img_min = 0.0		global_img_max = 19941.026744724255
global_T_min = 299.0		global_T_max = 2828.0
global_fv_min = 0.0		global_fv_max = 11.224797513519933
Fvmax_height = 808		Fvmax_width = 213
Imagemax_height = 808	Imagemax_width = 213
input_shape = (3, 808, 213)	output_shape = (808, 213)
setImgValZero = 0
setFvValZero = 0.01
setTValZero = 1000.0
~~~~~Params for model training~~~~~
model_name = CNNencdec
batch_size = 12
criterion = {loss_fv = F.mse_loss(outputs[:, 0, :, :], gts[:, 0, :, :])
                  loss_T = F.mse_loss(outputs[:, 1, :, :], gts[:, 1, :, :])
                  loss = 0.5 * loss_fv + 0.5 * loss_T}
lr = 0.0001
num_epochs = 300, early_stop_patience = 15
device = cuda
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
                   (self.config.optimizer, mode='min', factor=0.3, patience=3)
optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
```

---

### Dataset Creation and Data Manipulations

1. **Synthetic Data** was generated (see [linked repo](https://github.com/MaamIL/SootProfilesGenerateSyntheticData)).
2. Extracted out of *.mat files only valid image ranges (removed those with image values greater than 20000 or smaller than 0).
3. global min/max values were calculated throughout the dataset.
4. **Preprocessing**:
   - Flip image vertically (if needed- so the wider base will be on top)
   - Reset small values (configurable) to 0 (`< 0.1` for Fv, `< 1000` for T)
   - Normalize with min/max of dataset

      ```python
      fv = (fv - fv_global_min) / max((fv_global_max - fv_global_min), 1e-6)
      T = (T - T_global_min) / max((T_global_max - T_global_min), 1e-6)
      img = (img - img_global_min) / max((img_global_max - img_global_min), 1e-6)
      ```
    - Pad all image samples with 0.0 to trained data size- `808x213` (dataset’s maximum shape)

5. **Splits**:
Data was wrapped into train_loader (70%), val_loader (20%), test_loader (10%).
```
Train: 8307
Validation: 2375
Test: 1186
Total: 11868 (synthetic samples)
```

6. **Postprocessing (for test only)**: calculate the normalized value of the configured small values (see 4.2 above) and reset them to 0 before error analysis.

---

### Train Results

- Average epoch: Train ~6.5 min, Validation ~1 min
- Best model saved at `epoch 32`
- Validation loss: `0.00010501`
- Test loss: `0.00010567`
- 
<div align="center">
  <img src="pics for documentation\loss.png" alt="loss">
</div>

**Validation Visuals per Epoch (plotted for 4 samples)**:
- Flame input image (normalized and flipped back)
- Flame original image (from CFDImage.mat)
- Fv GT (normalized) heatmap
- Fv GT (normalized) CSV
- T GT (normalized) heatmap
- T GT (normalized) CSV
- Fv Prediction heatmap (every 10 epochs)
- T Prediction heatmap (every 10 epochs)
- Fv and T Error heatmaps and current loss (fixed and relative scales for better analysis ability)

**Test Visuals (plotted for 10 samples)**:
- Flame input image (normalized and flipped back)
- Flame original image (from CFDImage.mat)
- Fv GT (normalized) heatmap
- Fv GT (normalized) CSV
- T GT (normalized) heatmap
- T GT (normalized) CSV
- Fv Prediction heatmap 
- T Prediction heatmap
- Fv Prediction CSV 
- T Prediction CSV 
- Fv and T Error heatmaps and current loss (fixed and relative scales for better analysis ability)

*Validation sample results through the epochs*
<div align="center">
  <img src="pics for documentation\Val8813.jpg" alt="8813">
</div>  

*Test sample results*
<div align="center">
  <img src="pics for documentation\TEST7025.png" alt="7025">
</div>   


Another view of this test sample results- 3 cross sections in different heights (1, 300, 600), comparing the GT in green and the prediction in orange for each height:
<div align="center">
  <img src="pics for documentation\FvpredgtTEST7025.png" alt="7025fv">
</div>  
<div align="center">
  <img src="pics for documentation\TpredgtTEST7025.png" alt="7025t">
</div>  

---

### Inference and Single Tests

All training samples are padded to match the largest image in the dataset. Therefore I need to use only padding. 
However, when taking a sample out from the training dataset- it might be larger. In this case, I first crop the marginal rows and columns that hold only zeros. If still to big- I crop from the right and the bottom (reminder- the bottom of the image is the tip of the flame due to the flip process).
When running a single sample:
- If only image: run in **inference mode** (predict only).
- If image + GTs: run in **single test mode** (predict and calculate loss, plot results).

**Single test on synthetic sample (not in training dataset)**  
<div align="center">
  <img src="pics for documentation\SingleTest_58.png" alt="58">
</div>  

**Single test on real sample**  
Real data may be slightly different than synthetic data. Notice the output is different than the data the model was trained on. However- the cross sections show that in the meaningful parts (in the middle)- the results are not bad:
<div align="center">
  <img src="pics for documentation\SingleTest_realData0.png" alt="realdata">
</div> 
<div align="center">
  <img src="pics for documentation\SingleTest_realData0T.png" alt="realdataT">
</div> 
<div align="center">
  <img src="pics for documentation\SingleTest_realData0Fv.png" alt="realdataFv">
</div> 

**Inference on real data**  
3 samples:
<div align="center">
  <img src="pics for documentation\realdata1Infer.png" alt="realdataInfer1">
</div> 

<div align="center">
  <img src="pics for documentation\realdata2Infer.png" alt="realdataInfer2">
</div> 

<div align="center">
  <img src="pics for documentation\realdata3Infer.png" alt="realdataInfer3">
</div> 



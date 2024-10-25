<div align="center">

# **SolarFusionNet: Enhanced Solar Radiation Forecasting via Automated Multi-Modal Feature Selection and Cross-Modal Fusion**
[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_1.3+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.2-89b8cd)](https://hydra.cc/) 
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](#license)
</div>

## Description

This is the official repository to the paper ["Enhanced Regional Solar Radiation Forecasting via Automated Multi-Modal Feature Selection and Cross-Modal Fusion"](https://ieeexplore.ieee.org/document/10723760). In this research work, we propose a novel deep learning framework called Fusionformer - a novel attention-based deep learning architecture that effectively integrates automatic multi-modal feature selection and cross-modal fusion for enhancing regional solar radiation forecasting. 
Fusionformer utilizes two distinct types of automatic variable feature selection units to extract relevant features from multichannel satellite imagery and multivariate meteorological data, respectively. Long-term dependencies are then captured using three types of recurrent layers, each tailored to the corresponding modal data type. In particular, a novel Gaussian kernel-injected convolutional long short-term memory network is specifically designed to isolate the sparse features present in optical flow. Subsequently, a multi-head cross-modal self-attention mechanism is introduced to investigate the coupling correlation between the three modalities. The experimental results indicate that Fusionformer exhibits robust performance in predicting regional solar irradiance, 
achieving forecast skill of 47.6\% compared to the smart persistence model for the 4-hour-ahead forecast.
<div align="center">
<img src="pictures/framework.png" width="550">
</div>

## Dataset

### Satellite
The link to download the satellite data is [EUMETSAT](https://console.cloud.google.com/marketplace/product/bigquery-public-data/eumetsat-seviri-rss?hl=en-GB&project=triple-shadow-397515). Select the RSS dataset. Then use "reproject.py" in the scripts to cut the region, you need to set up the yaml file in configs before cutting. 

### Meteorological data
BSRN data can be downloaded by referring to the [Solar data](https://github.com/dazhiyang/SolarData) provided by Prof. Yang, or by directly visiting the [BSRN official website](https://bsrn.awi.de/). Don't forget to perform quality control.


## Training process
The pytorch modules required for the model must be installed before starting to train the model, detailed versions can be found in [requirements.txt](requirements.txt). After the requirements for model training have been met, all the files in configs need to be configured.


## Installation

#### Pip
```bash
# clone project
https://github.com/JOTYtao/Solar_Fusionformer.git
# create conda environment
conda create -n Solar_Fusionformer python=3.9
conda activate Solar_Fusionformer
# install requirements
pip install -r requirements.txt
```

## License

SolarFusionNet is licensed under the MIT License.

```
MIT License

Copyright (c) 2024 JOTYtao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

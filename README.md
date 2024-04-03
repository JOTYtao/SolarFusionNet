<div align="center">

# **Enhanced Regional Solar Radiation Forecasting via Automated Multi-Modal Feature Selection and Cross-Modal Fusion**

</div>

## Description

This is the official repository to the paper "Enhanced Regional Solar Radiation Forecasting via Automated Multi-Modal Feature Selection and Cross-Modal Fusion". In this research work, we propose a novel deep learning framework called Fusionformer - a novel attention-based deep learning architecture that effectively integrates automatic multi-modal feature selection and cross-modal fusion for enhancing regional solar radiation forecasting. 
Fusionformer utilizes two distinct types of automatic variable feature selection units to extract relevant features from multichannel satellite imagery and multivariate meteorological data, respectively. Long-term dependencies are then captured using three types of recurrent layers, each tailored to the corresponding modal data type. In particular, a novel Gaussian kernel-injected convolutional long short-term memory network is specifically designed to isolate the sparse features present in optical flow. Subsequently, a multi-head cross-modal self-attention mechanism is introduced to investigate the coupling correlation between the three modalities. The experimental results indicate that Fusionformer exhibits robust performance in predicting regional solar irradiance, 
achieving forecast skill of 47.6\% compared to the smart persistence model for the 4-hour-ahead forecast.

<img src="pictures/framework.png" width="350">

## Dataset
### Satellite
The link to download the satellite data is [EUMETSAT](https://console.cloud.google.com/marketplace/product/bigquery-public-data/eumetsat-seviri-rss?hl=en-GB&project=triple-shadow-397515). Select the RSS dataset. Then use "reproject.py" in the scripts to cut the region, you need to set up the yaml file in configs before cutting. 

### Meteorological data
BSRN data can be downloaded by referring to the [Solar data](https://github.com/dazhiyang/SolarData) provided by Prof. Yang, or by directly visiting the [BSRN official website](https://bsrn.awi.de/). Don't forget to perform quality control.


## Training process
The pytorch modules required for the model must be installed before starting to train the model, detailed versions can be found in [requirements.txt](requirements.txt). After the requirements for model training have been met, all the files in configs need to be configured.

## License

CrossViVit is licensed under the MIT License.

```
MIT License

Copyright (c) (2024) Ghait Boukachab

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

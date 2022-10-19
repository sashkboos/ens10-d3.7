
![Logo](https://raw.githubusercontent.com/spcl/ens10/main/figures/post-processing_pipeline-1.png?raw=true)

# ENS-10: A Dataset for Ensmeble Post-Processing.

This repository contains the instructions and examples for using the [ENS-10 dataset](https://arxiv.org/abs/2206.14786). 


> **Abstract:** Post-processing ensemble prediction systems can improve weather forecasting, especially for extreme event prediction.
In recent years, different machine learning models have been developed to improve the quality of the post-processing step. However, these models heavily rely on the data and generating such ensemble members requires multiple runs of numerical weather prediction models, at high computational cost. 
This paper introduces the ENS-10 dataset, consisting of ten ensemble members spread over 20 years (1998--2017). The ensemble members are generated by perturbing numerical weather simulations to capture the chaotic behavior of the Earth. 
To represent the three-dimensional state of the atmosphere, ENS-10 provides the most relevant atmospheric variables in 11 distinct pressure levels as well as the surface at 0.5-degree resolution.
The dataset targets the prediction correction task at 48-hour lead time, which is essentially improving the forecast quality by removing the biases of the ensemble members. To this end, ENS-10 provides the weather variables for forecast lead times T=0, 24, and 48 hours (two data points per week). We provide a set of baselines for this task on ENS-10 and compare their performance in correcting the prediction of different weather variables. We also assess our baselines for predicting extreme events using our dataset. The ENS-10 dataset is available under the Creative Commons Attribution 4.0 International (CC BY 4.0) licence.

For any questions, please create an issue. 

## Leaderboard


| Model | Z500 | T850 | T2m | Note  | Reference |
|:-:|:-:|:-:|:-:|-|--------------------------------------------------------|
| LeNet-Style |**74.41±0.109**|0.674±2e−4|0.659±4e−4| Following [Li, Wentao, et al.](https://www.sciencedirect.com/science/article/pii/S0022169421013512) | [Ashkboos, Saleh, et al. 2022](https://arxiv.org/abs/2206.14786) |
| U-Net |76.25±0.106| 0.669±0.009 |0.644±0.006| Following [Grönquist, Peter, et al.](https://spcl.inf.ethz.ch/Publications/.pdf/rsta-weather-postproc.pdf) | [Ashkboos, Saleh, et al. 2022](https://arxiv.org/abs/2206.14786) |
| Transformer |74.79±0.118|**0.665±0.002**|**0.626±0.004**| Following  [Finn, Tobias Sebastian](https://arxiv.org/pdf/2106.13924.pdf) | [Ashkboos, Saleh, et al. 2022](https://arxiv.org/abs/2206.14786)  |

*To add a new record to the leaderboard, please create an issue with your code as well as your results on *Z500*, *T850*, and *T2m* variables. All the experiments should be run with three different random seeds and the issue should containt the mean and standard deviation of each experiment. 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Usage


### CliMetLab plugin

CliMetLab is a Python package to simplify accessing meteorological and climate datasets. ENS-10 can be downloaded using a [CliMetLab Plugin](https://github.com/spcl/climetlab-maelstrom-ens10). 

For a fixed date, the data points can be accessed for both surface-level data, and pressure-level data for above-ground forecasts using a few lines of code:

```python
!pip install climetlab climetlab-maelstrom-ens10
import climetlab as cml

# Pressure-level data
ds = cml.load_dataset("maelstrom-ens10", date='20170226', dtype='pl')

# Surface-level data
ds = cml.load_dataset("maelstrom-ens10", date='20170226', dtype='sfc')

# Alternatively, the year can be omitted, and pressure levels are given by default:
# ds = cml.load_dataset("maelstrom-ens10", date='0226')

# Convert dataset to xarray data
ds.to_xarray()
```

For demo notebooks, see [here](https://github.com/spcl/climetlab-maelstrom-ens10/tree/main/notebooks).



### Direct Download
The dataset is hosted on the [ECMWF servers](https://storage.ecmwf.europeanweather.cloud/MAELSTROM_AP4/). All files can be downloaded alternatively using [this](http://spclstorage.inf.ethz.ch/projects/deep-weather/ENS10/) link. All files have `2018` prefix.
 

### Ground Truth Data

We use [ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=form) as the ground truth data in our baselines. The data we used in our baselines can be downloaded using [this](http://spclstorage.inf.ethz.ch/projects/deep-weather/ERA5/) link.

### Extreme Forecast Index (EFI)

The pre-computed EFI values (for `T2m`, `Z500`, and `T850` variables) over the test set (2016-2017 years) are available [here](http://storage.spcl.inf.ethz.ch/projects/deep-weather/ENS10/EFI/). 
We also provide a set of scripts to extract the EFI [here](https://github.com/spcl/ens10/tree/main/EFI).  


### Train Baseline Models

To train the baseline model(s) in the paper, run this command:

```train
python Train.py --model <model_name> --data-path <path_to_data> --target-var <predicted_variable>
```



 
##  Structure

* `baselines` -- this folder contains all scripts for running the baseline models.
* `baselines/utils` -- this folder contains all scripts for extracting the data, converting from GRIB to Numpy, and metric.
* `EFI` -- this folder contains all scripts for extracting the extreme forecast index over the dataset.



##  License

The ENS-10 dataset is available under the Creative Commons Attribution 4.0 International (CC BY 4.0) licence (see [here](https://github.com/spcl/ens10/blob/main/LICENSE)).


##  How to cite

```
@article{ashkboos2022ens,
  title={ENS-10: A Dataset For Post-Processing Ensemble Weather Forecast},
  author={Ashkboos, Saleh and Huang, Langwen and Dryden, Nikoli and Ben-Nun, Tal and Dueben, Peter and Gianinazzi, Lukas and Kummer, Luca and Hoefler, Torsten},
  journal={arXiv preprint arXiv:2206.14786},
  year={2022}
}
```

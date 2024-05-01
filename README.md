# double-wilson

Exploring the theoretical basis for the double-Wilson distribution

## Installation

Before notebooks can be run, the environment must be set up. Please run the following commands:

```
git clone https://github.com/Hekstra-Lab/dw.git
cd dw
conda env create --file wilson.yml
conda activate wilson
pip install -e .
```

## Summary
This repository contains a set of IPython notebooks, each standalone and exploring a separate idea around the theoretical basis of the double-Wilson distribution. The notebooks are as follows:
- [Notebook 1: Normalizing structure factor amplitudes](1_Dataset_prep_and_local_scaling.ipynb): A notebook using multiple local scaling methods to prepare normalized structure factor amplitudes from real data, along with an accompanying notebook, [Notebook 1A: Normalizing structure factor amplitudes for anomalous data](1A_Anom_dataset_prep_and_scaling.ipynb) for doing the same with anomalous data. 
- [Notebook 2a: Illustrating the double-Wilson model using synthetic data](2a_Synthetic_data_example.ipynb): A notebook illustrating the double-Wilson model using synthetic data. 
- [Notebook 2b: The double-Wilson model with measurement error and resolution-dependence](2b_Measurement_error_res_dependence.ipynb): A notebook exploring ways to incorporate measurement error and resolution dependence into the double-Wilson model. 
- [Notebook 3: Fitting a double-Wilson model to a pair of datasets](3_Fitting_DW_to_paired_data.ipynb): A notebook that fits paired data scaled in [Notebook 1: Dataset preparation and local scaling](1_Dataset_prep_and_local_scaling.ipynb). This is accompanied by [Notebook 3A: Fitting a Double-Wilson model to an anomalous dataset](3A_Fitting_the_DW_model_to_anomalous_data.ipynb). 
- [Notebook 4: Conventions for parametrizing the Double-Wilson distribution](4_Parsing_DW_parameters.ipynb): A small notebook for converting between notational conventions for the Rice and folded Normal probability distributions.
- [Notebook 5: A model for bivariate priors](5_Bivariate_priors.ipynb): A notebook containing derivations of important probability distributions in the double-Wilson model. 
- [Notebook 6: Global Scaling for Difference Maps](6_Revisiting_difference_maps.ipynb): A notebook exploring a formalism for difference maps under the double-Wilson model. 
- [Notebook 7: The general case: any number of sets of related structure factors](7_DAG_PGM_structure_factor_distributions.ipynb): A notebook constructing a formalism that relates more than three sets of structure factor amplitudes in the multivariate Wilson model.

These notebooks rely on files in the following folders:
- `dag_img`: a folder containing images of directed acyclic graphs, introduced in [Notebook 7]((7_DAG_PGM_structure_factor_distributions.ipynb). 
- `double_wilson_data`: a folder containing MTZ files containing paired datasets for use in [Notebook 1](1_Dataset_prep_and_local_scaling.ipynb), [Notebook 1A](1A_Anom_dataset_prep_and_scaling.ipynb), [Notebook 2b](2b_Measurement_error_res_dependence.ipynb), [Notebook 3](3_Fitting_DW_to_paired_data.ipynb), and [Notebook 3A](3A_Fitting_the_DW_model_to_anomalous_data.ipynb). 
- `dw_tools`: a python package containing tools used in all notebooks. 
- `results_figs`: a folder containing figure outputs from most notebooks. 
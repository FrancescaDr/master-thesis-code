# Predicting immune responses on multi-modal single-cell data with variational inference

This code accompanies the master thesis `Predicting immune responses on multi-modal single-cell data with variational inference` (https://repository.tudelft.nl/islandora/object/uuid%3A1b24699a-3967-4b08-9316-dae8d9577222?collection=education). 

Author: Francesca Drummer

Supervisors: Dr. Ahmed Mahfouz and Mikhael Manurung

## Package Structure

The repository is centered around the scr\_trainer module:

- `src\_trainer.main` contains training and evaluation functions 
- `src\_trainer.preprocessing` contains data preprocessing 
- `src\_trainer.plotting` contains ModelEvaluation class and functions for plotting
- `src\_trainer.SCVI\_model` contains scVI model trained with RNA 
- `src\_trainer.TOTALVI\_model` contains totalVI model 
- `src\_trainer.cellPMVI\_model` contains variants of cellPMVI model:
	- `cellPMVI` with isotropic normal prior (uses `cellPMVAE` module)
	- `cellPMVI\_lp` with Laplace prior (uses `cellPMVAE\_lp` module)
- `src\_trainer.cellPMVI\_CITESEQ` contains adaption of cellPMVI model that is based on totalVI (uses `cellPMVVAE\_CITESEQ` module)
- `src\_trainer.my\_base\_component` contains cellPMVI encoder variant
- `src\_trainer.my\_training\_plan` contains own extension of training plan
- `src\_trainer.my\_vae` contains cellPMVI VAE variant

Additional files and folders:

- `notebooks` contains notebooks to reproduce plots from the paper and detailed analysis of each model
- `scripts` contains the bash file for automatic running of the model
- `CPA` necessary adjustments to CPA to run with czi data
- `input` contains trained models
- `diff_exp` contains each cell types csv file with p-value of the differential expression analysis
- `data` contains datasets in h5ap format
- `results` contains the csv and pickle files after model evaluation 

## Run

There are two options for executing the main file: 1) Training and 2) Evalution of a trained model. 
The first argument `--func` defines which of them gets executed: 

1. `--func train\_model`

2. `--func evaluate\_model`

### Training

Mandatory arguments

- `--dataset\_path`: Respective location of .h5ad data to load
- `--model\_type`: Type of model to train. There are four different available types of models: 
	- `SCVI\_RNA`: scvi model with RNA data
	- `SCVI\_protein`: scvi model with protein data
	- `MMVAE`: MMVAE model with one encoder for each RNA and protein 
	- `TotalVI`: default TotalVI model from scvi-tools

## Evaluation

Mandatory arguments:

- `--filename`: model name (DATE combination)
- `--model\_type`: Type of model to evaluate 
- `--training\_scenario`: Training scenario 1,2, or 3 for evaluation 





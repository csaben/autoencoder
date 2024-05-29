# autoencoder

sample autoencoder implementation

## Goal
- Implement an autoencoder to reconstruct non anomalous data
- use latent emedding space to + MSE to determine anomalies


## Data
- [source NIH](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html)
- [train (uninfected)](https://data.lhncbc.nlm.nih.gov/public/Malaria/NIH-NLM-ThickBloodSmearsU/NIH-NLM-ThickBloodSmearsU.zip)
- [anomalies (infected)](https://data.lhncbc.nlm.nih.gov/public/Malaria/NIH-NLM-ThickBloodSmearsPV/NIH-NLM-ThickBloodSmearsPV.zip)

## Instructions
Download and extract data into a folder `data` in the root of the project

## Installation

### Inside current `venv`

```bash
python -m pip install -e .
```

### Inside new `venv`
```bash
conda create -n autoencoder python=3.10
conda activate autoencoder
python -m pip install -e .
```



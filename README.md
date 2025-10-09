# SNAKE-fMRI: Simulator from neuro-activation to K-space Exploration


<img align="left" width="33%" src="https://github.com/paquiteau/snake-fmri/blob/main/docs/_static/logos/snake-fmriV2-logo.png"> 
<h1> A Simulator from Neurovascular coupling to Acquisition of K-space data for Exploration of fMRI Technique </br></br></h1>


[![Test](https://github.com/paquiteau/snake-fmri/actions/workflows/test.yml/badge.svg)](https://github.com/paquiteau/snake-fmri/actions/workflows/test.yml)
[![deploy-docs](https://github.com/paquiteau/snake-fmri/actions/workflows/deploy-docs.yml/badge.svg)](https://paquiteau.github.io/snake-fmri)
[![HAL](https://img.shields.io/badge/preprint-04533862-purple?logo=HAL&logoColor=white)](https://hal.science/hal-04533862)

[![python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=blue)](https://pypi.org/project/snake-fmri)
![black](https://img.shields.io/badge/code--style-black-black)
![ruff](https://img.shields.io/badge/lint-ruff-purple?logo=stackblitz&logoColor=yellow)



This package provides a simulation and reconstruction framework for fMRI data. It is designed to be used for benchmarking and testing of fMRI reconstruction methods.


## Installation
### Requirements 
- A working Python 3.10 environment  or higher 
- Optional: a working CUDA environment for NUFFT accelerations. See [mri-nufft](https://github.com/mind-inria/mri-nufft) for details. SNAKE works best with a fast GPU NUFFT Backend such as [cufinufft](https://github.com/flatironinstitute/finufft) or [gpuNUFFT](https://github.com/chaithyagr/gpuNUFFT)


### from PyPA (soon)

It is recommended to install the SNAKE toolkit on top of the core runtime (the toolkit contains reconstructors, statistical analysis tool and the CLI to run experiments).

``` sh
pip install snake-fmri[toolkit]
```

Some Reconstructors requires extra dependencies such as [pysap-fmri](https://github.com/paquiteau/pysap-fmri) or [patch-denoising](https://github.com/paquiteau/patch-denoising)


### Development version 

``` sh
git clone git@github.com:paquiteau/snake-fmri 
cd snake-fmri 
pip install -e .[test,dev,doc,toolkit]
```


## Getting Started

Documentation is available at https://mind-inria.github.io/snake-fmri/

To get started, you can check the examples gallery: https://paquiteau/github.io/snake-fmri/examples

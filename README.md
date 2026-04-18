# LLMsafe-Spectre
An intelligent HPC event selection method-ology designed for efficient Spectre attack detection in LLM
environments

## Overview

Existing HPC-based Spectre detectors perform well in conventional or low-noise environments, but their accuracy degrades significantly when concurrent LLM workloads are introduced, especially for **locally deployed large-scale LLMs**.  
LLMsafe-Spectre addresses this problem through a **two-stage intelligent HPC event selection pipeline**:

1. **LLM-instructed analytical filtering**
   - Semantic clustering of HPC events  
   - Redundancy elimination  
   - Multi-LLM union of candidate events  

2. **ANOVA-based statistical filtering**
   - Inter-group and intra-group variance analysis  
   - Top-K event ranking and selection  

This design enables efficient and robust Spectre detection while respecting the limited number of hardware PMU registers available on modern processors.

## Top 5 feature
The top five feature values under three CPU architectures (cascadelake，skylake，broadwell) have been uploaded.

## Source Code
The code files are available in the code directory of the repository. 
Source code has been uploaded to the `code` directory of this repository, including: training and testing code, data collection code, feature selection and evaluation scripts.



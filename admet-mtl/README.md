# ADMET Prediction with Multi-Task Learning

## Overview
This project focuses on predicting ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) properties of chemical compounds using machine learning.

We compare:
- Single-task models (one model per endpoint)
- Multi-task learning (shared representation across endpoints)

## Objectives
- Evaluate whether multi-task learning improves prediction quality
- Compare different molecular representations
- Analyze which ADMET endpoints benefit from shared learning

## Dataset
We use datasets from:
- Therapeutics Data Commons (TDC)

Example endpoints:
- Solubility
- Caco-2 permeability
- hERG toxicity
- Clearance

## Project Structure
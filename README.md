# RBG Performance

This repository contains machine learning experiments using the RBG (Regressão Bem Geral) library, developed in a companion repository, and two public datasets: Rent99 and Dry Bean.

The main objective of this project is to analyze and evaluate the impressive performance of the RBG through its application to different machine learning problems. The experiments focus on exploratory data analysis, data preprocessing, model training, cross-validation, hyperparameter tuning, model comparison, and evaluation.

## Datasets

### Rent99
A regression dataset containing information about rental properties in Munich in 1999. The objective is to study and predict monthly rental prices.

Source: https://rdrr.io/cran/gamlss.data/man/rent99.html

### Dry Bean
A classification dataset containing 13,611 observations of seven different dry bean varieties, described by 16 numerical features.

Source: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

## Repository Structure

- `notebooks/`: exploratory data analysis and model experiments.
- `data/`: information about the datasets and their sources.
- `reports/`: final results, tables, visualizations, and conclusions.
- `docs/`: experimental methodology and RBG integration documentation.

## Experimental Workflow

1. Exploratory Data Analysis (EDA).
2. Data preparation and preprocessing.
3. Train-test split.
4. Cross-validation.
5. Hyperparameter tuning.
6. Model training and comparison.
7. Final evaluation.
8. Interpretation of results, focusing on RBG's performance.

## Requirements

Install the project dependencies using:

```bash
pip install -r requirements.txt
```

See `docs/rbg_integration.md` for information about installing and using the RBG library.

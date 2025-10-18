# Credit Approval Classifier

A machine learning project that predicts whether a credit application should be approved based on applicant data. The project includes both Python and R scripts for model training, evaluation, and analysis.

## Overview
This project explores data preprocessing, feature selection, and classification techniques for predicting credit approval outcomes. It compares different models and evaluates their performance using common metrics such as accuracy and precision.

## Features
- Data cleaning and preprocessing  
- Feature encoding and scaling  
- Training and testing multiple classification models  
- Model evaluation and comparison  
- Implemented in both Python and R

## Files
```
Credit-Approval-Classifier/
├── Credit_Analysis.py      # Python version of the analysis
├── Credit_Analysis.R       # R version of the analysis
├── credit+approval/        # Dataset and related files
└── Econometrics.pdf        # Report summarizing results
```

## How to Run (Python)
1. Install Python 3.10+  
2. Install required packages:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Run the script:
   ```bash
   python Credit_Analysis.py
   ```

## How to Run (R)
1. Open `Credit_Analysis.R` in RStudio.  
2. Install required libraries (e.g., tidyverse, caret).  
3. Run all cells or source the script:
   ```r
   source("Credit_Analysis.R")
   ```

## Results
The scripts output model performance metrics and visualizations showing how different algorithms perform on the credit approval dataset.

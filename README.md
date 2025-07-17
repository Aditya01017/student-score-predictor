# Student Performance Prediction Project

## Project Overview
This project predicts student final exam scores (G3) based on various factors like study time, previous grades, attendance, and other student characteristics. The goal is to help educators identify students who might need additional support.

## Files Description

### Notebooks (Run in order):
1. **01_data_exploration.ipynb** - Understanding the dataset and exploring relationships
2. **02_data_preprocessing.ipynb** - Cleaning and preparing data for modeling
3. **03_model_development.ipynb** - Training different regression models
4. **04_model_evaluation.ipynb** - Comparing models and analyzing results

### Data:
- **student-mat.csv** - Student dataset for Math course

### Other Files:
- **requirements.txt** - Python libraries needed
- **best_model.pkl** - Saved best performing model
- **results_summary.txt** - Summary of model performance and insights

## How to Run

1. **Install libraries:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run notebooks in order:**
   - Open Jupyter: `jupyter notebook`
   - Run each notebook from 01 to 04

## Key Features

### Data Cleaning:
- Handle missing values
- Encode categorical variables
- Normalize numerical features

### Models Used:
- Linear Regression
- Random Forest
- XGBoost

### Evaluation Metrics:
- Mean Squared Error (MSE)
- R² Score
- Cross-validation

### Key Insights:
- Study time is the most important predictor
- Previous grades (G1, G2) strongly influence final grades
- Attendance has moderate impact on performance

## Project Results
- **Best Model:** Random Forest
- **R² Score:** ~90%
- **Key Factors:** Study time, previous grades, attendance

## Learning Objectives
- Understand data science workflow
- Learn regression techniques
- Practice data preprocessing
- Interpret model results
- Generate business insights 
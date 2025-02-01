# oibsip_task2

# Task 2: 

Unemployment Analysis

# Task Overview:

--> This project focuses on analyzing unemployment trends in India using Python. The goal is to gain insights into the unemployment rates by visualizing and 
    interpreting the provided data. This analysis will help understand the impact of unemployment in various regions of the country.

The dataset used in this project contains information about unemployment rates across different states in India, categorized by regions. It also includes dates and other relevant economic factors.

# Objectives:

Data Cleaning: Handle missing values and ensure the dataset is ready for analysis.

Exploratory Data Analysis (EDA): Generate insights through visualizations, such as line plots, bar charts, and heatmaps.

Regional Comparison: Analyze unemployment rates across different regions of India.

Time Series Analysis: Study the trend of unemployment over time.

# Project Structure:

Task_2_Unemployment_Analysis/

├── data/

│   └── Unemployment_in_India.csv  # Original dataset

├── outputs/

│   └── Data_recorded.png

│   └── cleaned_unemployment_data.csv  # Cleaned dataset saved by the script

├── src/

│   └── unemployment_analysis_jupyter.py

│   └── unemployment_analysis_py.py  # Python script for analysis

├── visualizations/

│   └── Corelation_heatmap.png

│   └── Distribution_UR.png

│   └── Regional_rates.png

│   └── Unemployment_rate.png

├── requirements.txt                # Dependencies
             

# Installation:

Prerequisites:

Make sure you have Python 3.7 or later installed. The required libraries are listed in the requirements.txt file.

# Steps to Install:

Clone the repository:

git clone https://github.com/your-username/your-repo-name.git
cd Task_2_Unemployment_Analysis

Install the dependencies:

pip install -r requirements.txt

# Usage:

Run the analysis script:

python src/unemployment_analysis.py

This will:

Clean the dataset.

Perform EDA and save the visualizations.

Save the cleaned dataset to the outputs/ folder.

# Access the results:

View the generated visualizations in the visualizations/ folder.

The cleaned dataset can be found in the outputs/ folder.

# Dataset:

The dataset used for this project is included in the data/ folder:

File: Unemployment_in_India.csv

Description: Contains unemployment data across different regions in India, along with dates and other economic indicators.

# Key Visualizations:

Unemployment Trends Over Time:

Line charts to observe how unemployment rates have changed over the years.

Regional Comparison:

Bar charts and heatmaps to compare unemployment rates across different regions.

Correlation Analysis:

Heatmaps to identify relationships between various economic factors.

# Requirements:

The following Python libraries are required for this project:

pandas
numpy
matplotlib
seaborn

# Install them by running:

pip install -r requirements.txt

# Results and Insights:

The analysis revealed significant variations in unemployment rates across regions.

Time series analysis identified trends and periods of high unemployment.

The visualizations provided clear insights into the data, highlighting the most affected regions.

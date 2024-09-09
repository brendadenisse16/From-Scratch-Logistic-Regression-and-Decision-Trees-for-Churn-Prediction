# Customer Churn Prediction using Statistical Learning

## Overview
This project focuses on predicting customer churn for bank customers using statistical learning techniques. Customer retention is crucial for organizational success, especially in the financial sector. The data used in this project comes from an actual bank dataset obtained via [Kaggle](https://www.kaggle.com/).

The goal is to build a predictive model that helps banks identify which customers are likely to churn, allowing them to implement proactive retention strategies.

## Algorithms Used
I implemented two statistical learning algorithms **from scratch** in R, without using any built-in machine learning functions:
1. **Logistic Regression**: A widely used algorithm for binary classification problems.
2. **Decision Trees**: A model that provides high interpretability and is particularly useful for understanding customer behavior.

These models were fully coded from the ground up, showcasing the detailed mathematical understanding and coding skills behind each technique.

## Performance Metrics
To evaluate the models, we used the **Area Under the Receiver Operating Characteristic (AUROC)** curve:
- Logistic Regression performed better overall.
- Decision Trees offered higher interpretability, making them useful for marketing strategy insights.

## Results
- **Logistic Regression**: Achieved an AUROC of 0.87, indicating a strong ability to distinguish between churners and non-churners.
- **Decision Tree**: Achieved an AUROC of 0.79 but provided valuable insights into customer behavior patterns.

## Key Feature: Code from Scratch
One of the key contributions of this project is that all algorithms were developed from scratch. This provides a deeper understanding of the mechanics behind Logistic Regression and Decision Trees, as well as the coding techniques required to implement these models without relying on external machine learning libraries.

## Project Structure
- **/BinaryTreeDecision.R**: R script implementing the Decision Tree model from scratch.
- **/LogisticRegression.R**: R script implementing Logistic Regression from scratch.
- **/churn.csv**: The dataset used for training and testing the models.

## Setup and Usage
To run the project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/brendadenisse16/From-Scratch-Logistic-Regression-and-Decision-Trees-for-Churn-Prediction.git
2. **Install the required packages**:
   The project requires R but does not rely on built-in machine learning libraries. The code uses basic R libraries for handling data:

   - **rpart** for simple decision-making logic (not model building).
   - **glm** is not used here; the logistic regression was coded from scratch.
3. **Run the R scripts**:
   Open the .R scripts in your RStudio or R environment and run them to see the model outputs.

# Customer Churn Prediction and Analysis

<div align="center">
  <img src="./Label_Image.png" alt="Customer Churn" width="600px">
</div>

## 🧠 Project Overview

This project implements advanced machine learning techniques to predict customer churn behavior. By analyzing patterns in customer data, businesses can identify at-risk customers and take proactive steps to improve retention.

## 🚀 Key Features

- Comprehensive data preprocessing and cleaning pipeline
- Statistical feature analysis using T-Tests, and Chi-Squared Tests
- Advanced feature engineering to capture customer engagement patterns
- Multiple ML models comparison: XGBoost, Random Forest, and Logistic Regression
- Detailed performance evaluation with confusion matrices, ROC curves, and AUC metrics
- MLOps implementation for model deployment and monitoring

## 📊 Project Structure

The project follows a methodical approach across four key milestones:

### Milestone 1: Data Preprocessing

- Data cleaning and handling missing values
- Outlier detection and treatment using IQR and Box-Cox transformations
- Feature transformation and encoding for categorical variables
- Normalization and scaling of numerical features

### Milestone 2: Advanced Analysis & Feature Engineering

- Statistical analysis using T-Tests, and Chi-Squared Tests
- Feature selection through Random Forest importance and statistical significance
- Creation of derived features (engagement score, interaction frequency, etc.)
- Dimensionality reduction techniques to manage feature space

### Milestone 3: Model Development & Evaluation

- Implementation of multiple algorithms:
  - Random Forest
  - Logistic Regression
  - XGBoost
- Hyperparameter tuning using RandomizedSearchCV and GridSearchCV
- Comprehensive performance evaluation using accuracy, precision, recall, F1, and AUC
- Model comparison to select the champion model

### Milestone 4: MLOps & Deployment

- Developed a user-friendly web application using **Streamlit** for real-time customer churn prediction.
- Integrated the preprocessing pipeline into the app to ensure consistency with the training process.
- Saved and loaded the trained scaler and model using **pickle** for seamless deployment.
- Designed an interactive interface for users to input customer data and view predictions.
- Deployed the application locally with instructions for easy setup and execution.
- Prepared the project for future cloud deployment with modular and scalable code structure.

## 📈 Results

Our model comparison revealed that **XGBoost with Important Features** achieved the best performance:

- **Accuracy**: 95.09%
- **Precision**: 95.19%
- **Recall**: 94.94%
- **F1 Score**: 95.04%
- **AUC**: 98.50%

This outperformed both Random Forest and Logistic Regression across all key metrics.

## 🛠️ Technologies Used

- **Data Processing**: Pandas, NumPy, Scipy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: Scikit-learn, XGBoost, Random Forest, Logistic Regression
- **Statistical Analysis**: T-Tests, Chi-Squared Tests
- **Deployment**: Streamlit, Pickle

## 🔧 Getting Started

1. Clone the repository:
   ```bash
   https://github.com/HazemAbdelghafar/Customer-Churn-DEPI
   cd Customer-Churn-DEPI
   ```

2. Install required dependencies:
   ```bash
    pip install -r requirements.txt
   ```
3. Run the Streamlit app:
    ```bash
      streamlit run app.py
    ```

4. Run the notebook cells to reproduce the analysis and model training.

<br>
<br>
<div align="center"> <b?><i>This project was developed as part of the DEPI graduation project.</i></b> </div>


# Customer Churn Prediction and Analysis

<div align="center">
  <img src="./Label_Image.png" alt="Customer Churn" width="600px">
</div>

## 🧠 Project Overview

This project implements advanced machine learning techniques to predict customer churn behavior. By analyzing patterns in customer data, businesses can identify at-risk customers and take proactive steps to improve retention.

## 🚀 Key Features

- Comprehensive data preprocessing and cleaning pipeline
- Statistical feature analysis using t-tests, ANOVA, and chi-squared tests
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

- Statistical analysis using t-tests, ANOVA, and chi-squared tests
- Feature selection through Random Forest importance and statistical significance
- Creation of derived features (engagement score, interaction frequency, etc.)
- Dimensional reduction techniques to manage feature space

### Milestone 3: Model Development & Evaluation

- Implementation of multiple algorithms:
  - Random Forest (with different feature subsets)
  - Logistic Regression
  - XGBoost
- Hyperparameter tuning using RandomizedSearchCV and GridSearchCV
- Comprehensive performance evaluation using accuracy, precision, recall, F1, and AUC
- Model comparison to select the champion model

### Milestone 4: MLOps & Deployment

- Model tracking and versioning with MLflow/DVC
- API development with Flask/FastAPI
- Cloud deployment strategies
- Performance monitoring and drift detection
- Automated retraining pipelines

## 📈 Results

Our model comparison revealed that **XGBoost with Important Features** achieved the best performance:

- **Accuracy**: 95.23%
- **Precision**: 94%
- **Recall**: 97%
- **F1 Score**: 96%
- **AUC**: 98.56%

This outperformed both Random Forest and Logistic Regression across all key metrics.

## 🛠️ Technologies Used

- **Data Processing**: Pandas, NumPy, Scipy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: Scikit-learn, XGBoost
- **Statistical Analysis**: t-tests, ANOVA, chi-squared tests
- **MLOps**: MLflow, Docker, Kubernetes (planned)

## 🔧 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/youssefaymanmohamed/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. Install required dependencies:
```bash
    pip install -r requirements.txt

```
4. Run the notebook cells to reproduce the analysis and model training.

<div align="center"> <i>This project was developed as part of the DEPI graduation project.</i> </div>


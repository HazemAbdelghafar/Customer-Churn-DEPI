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
- Dual deployment: Streamlit (interactive UI) and FastAPI (API + web interface)
- Logging for both API and Streamlit apps for monitoring and debugging

## 📁 Project Structure

```
Customer-Churn-DEPI/
│
├── app/                  # FastAPI web assets
│   ├── static/
│   │   └── style.css     # CSS for FastAPI web UI
│   └── templates/
│       └── index.html    # HTML for FastAPI web UI
│
├── model/                # Model artifacts
│   ├── best_xgb_model.pkl
│   └── scaler.pkl
│
├── fastapi_app.py        # FastAPI app (API + web interface)
├── streamlit_app.py      # Streamlit app (interactive UI)
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── ...                   # Other assets, notebooks, images
```

## 📊 Project Milestones

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
- Developed a robust **FastAPI** app for both API and web-based JSON input.
- Integrated the preprocessing pipeline into both apps to ensure consistency with the training process.
- Saved and loaded the trained scaler and model using **pickle** for seamless deployment (see `model/` directory).
- Designed interactive interfaces for users to input customer data and view predictions.
- Added logging to both apps for monitoring and debugging (`streamlit_app.log`, `fastapi_app.log`).
- Modular and scalable code structure for future cloud deployment.

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
- **Deployment**: Streamlit, FastAPI, Pickle, Logging

## 🔧 Getting Started

### 1. Clone the repository:
```bash
https://github.com/HazemAbdelghafar/Customer-Churn-DEPI
cd Customer-Churn-DEPI
```

### 2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app (interactive UI):
```bash
streamlit run streamlit_app.py
```

### 4. Run the FastAPI app (API + web interface):
```bash
uvicorn fastapi_app:app --reload --port 8000
```
- Visit [http://127.0.0.1:8000/](http://127.0.0.1:8000/) for the interactive JSON web interface.
- Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the interactive API documentation (Swagger UI).

### 5. Run the notebook cells to reproduce the analysis and model training.

### 6. Logging
- Streamlit logs: `streamlit_app.log`
- FastAPI logs: `fastapi_app.log`

<br>
<br>
<div align="center"> <b><i>This project was developed as part of the DEPI graduation project.</i></b> </div>


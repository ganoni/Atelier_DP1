
# Atelier 1

## Developed by: El MAhdi EL ATEKI GANONI
## Supervised by: Prof. EL ACHAAK Lotfi  

## Executive Summary
This project explores the application of deep learning in two key areas: predicting stock prices through regression and classifying data using a deep neural network. The goal is to highlight the versatility of deep learning in extracting valuable insights and developing predictive models across various domains. The project serves as a foundation for applying deep learning techniques to real-world financial and industrial challenges.

---

## Part 1: Stock Market Regression Analysis - Predicting Future Stock Prices

### Objective
Develop a deep learning model to predict stock closing prices based on historical market data, aiding investment decisions and risk management.

### Project Overview
This segment focuses on constructing a regression model to forecast stock prices. Key steps include data preprocessing, model training, evaluation, and visualization to assess performance and market trends.

### Dataset
- **Source**: Public dataset from Kaggle: [NYSE Dataset](https://www.kaggle.com/datasets/dgawlik/nyse)
- **Features**:
  - Date, Symbol, Open, Close (target variable), Low, High, Volume
- **Data Considerations**: Time-series relationships must be accounted for during modeling.

### Implementation Details
1. **Data Acquisition & Preprocessing**
   - Load stock data from a CSV file.
   - Handle missing values through imputation techniques (e.g., mean, median, interpolation).
   - Convert data types appropriately (e.g., date to datetime format).
   - Feature engineering:
     - Moving Averages (e.g., 5-day, 20-day)
     - Relative Strength Index (RSI)
     - Moving Average Convergence Divergence (MACD)
   - Apply feature scaling using StandardScaler or MinMaxScaler.

2. **Model Training**
   - Split data into training, validation, and test sets (e.g., 70%-15%-15%).
   - Define a deep learning model using TensorFlow or PyTorch:
     - Consider Multi-Layer Perceptron (MLP), Long Short-Term Memory (LSTM), or Gated Recurrent Units (GRU).
   - Use Mean Squared Error (MSE) as the loss function and Adam optimizer.
   - Implement early stopping to prevent overfitting.
   - Tune hyperparameters through grid search or random search.

3. **Evaluation**
   - Metrics: MSE, RMSE, R², MAE
   - Visualize predicted vs. actual prices and analyze residuals.

### Results & Discussion
- Summarize model performance with evaluation metrics.
- Identify strengths and weaknesses based on visualizations and error analysis.

### Potential Improvements
- Add more technical indicators (e.g., Bollinger Bands, Fibonacci Retracements).
- Incorporate sentiment analysis from news and social media.
- Experiment with advanced architectures (e.g., Transformers, ensemble models).
- Optimize hyperparameters using Bayesian optimization.
- Implement risk management strategies and backtesting.

---

## Part 2: Deep Learning Classification - Predictive Maintenance

### Objective
Build a deep learning model to classify machine conditions, predicting failures and optimizing maintenance schedules.

### Project Overview
This section focuses on developing a classification model for predictive maintenance, covering data processing, model training, evaluation, and interpretation.

### Dataset
- **Source**: Public dataset from Kaggle: [Predictive Maintenance Dataset](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)
- **Features**: Sensor readings, machine parameters, failure labels
- **Feature Selection**:
  - Correlation Analysis
  - Feature Importance (e.g., Random Forest, Gradient Boosting)
  - Univariate Feature Selection (e.g., chi-squared test, ANOVA)

### Implementation Details
1. **Data Acquisition & Preprocessing**
   - Load dataset and handle missing values.
   - Normalize data using StandardScaler or MinMaxScaler.
   - Split into training, validation, and test sets.

2. **Model Training**
   - Define a neural network architecture using TensorFlow/PyTorch:
     - Consider CNN, MLP, RNN, or hybrid models.
   - Use categorical cross-entropy loss and Adam optimizer.
   - Implement early stopping and dropout layers to prevent overfitting.

3. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC
   - Confusion matrix to analyze model predictions.

### Results & Discussion
- Provide a performance summary with key evaluation metrics.
- Identify areas for improvement based on confusion matrix insights.

### Potential Improvements
- Implement data augmentation techniques.
- Explore CNN architectures (e.g., ResNet, Inception).
- Optimize hyperparameters using grid search or Bayesian optimization.
- Use ensemble methods for better classification accuracy.
- Apply Explainable AI (XAI) techniques (e.g., SHAP, LIME).
- Incorporate cost-sensitive learning for better failure detection.

---

## Conclusion
This project demonstrates deep learning applications in stock market prediction and predictive maintenance classification. By implementing various preprocessing techniques, model architectures, and evaluation metrics, this work lays a strong foundation for further research. Future improvements can enhance model accuracy and real-world applicability, making deep learning an essential tool for finance and industry.

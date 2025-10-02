# 🍷Red-Wine-Quality-Prediction

This project predicts **wine quality** (good vs. bad) using machine learning algorithms on the **UCI Red Wine Quality dataset**.  
The goal is to compare multiple models and select the best performer based on accuracy.  

---

## 📂 Dataset  

- **Source**: [Wine Quality Data Set - UCI ML Repo](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)  
- **File Used**: `winequality-red.csv`  
- **Shape**: `1599 rows × 12 columns`  
- **Target Variable**:  
  - `quality` (scores 3–8)  
  - Converted into **binary classification**:  
    - `goodquality = 1` if quality ≥ 7  
    - `goodquality = 0` otherwise  

---

## 🛠️ Workflow  

1. **Data Preprocessing**  
   - Checked for missing values → none found  
   - Feature engineering: created `goodquality` label  
   - Exploratory Data Analysis (EDA)  
     - Boxplots, histograms, KDE plots  
     - Heatmap & correlation analysis  
     - Pairplots & violin plots  

2. **Feature Importance**  
   - Used `ExtraTreesClassifier` to find most important predictors.  
   - Top features: **alcohol, sulphates, volatile acidity**.  

3. **Model Training & Evaluation**  
   - Split data: 70% train, 30% test  
   - Models used:  
     - Logistic Regression  
     - K-Nearest Neighbors (KNN)  
     - Support Vector Classifier (SVC)  
     - Decision Tree  
     - Gaussian Naive Bayes  
     - Random Forest  
     - XGBoost  

---

## 📊 Results  

| Rank | Model                | Accuracy |
|------|----------------------|----------|
| 🥇   | Random Forest        | **0.894** |
| 🥈   | XGBoost              | 0.879    |
| 🥉   | KNN                  | 0.872    |
| 4    | Logistic Regression  | 0.870    |
| 5    | SVC                  | 0.868    |
| 6    | Decision Tree        | 0.865    |
| 7    | GaussianNB           | 0.833    |

👉 **Random Forest was selected as the final model** due to the highest accuracy.  

---

## 📦 Libraries Used  

- **NumPy, Pandas, Matplotlib, Seaborn** → data handling & visualization  
- **Scikit-learn** → ML models (Logistic Regression, KNN, SVC, Decision Tree, RF, Naive Bayes)  
- **XGBoost** → Gradient boosting classifier  

---

## 📌 Future Improvements  

- **Hyperparameter tuning** → GridSearchCV, RandomizedSearchCV  
- **Class imbalance handling** → SMOTE, class weights  
- **Advanced models** → LightGBM, CatBoost  
- **Cross-validation** → More robust evaluation  
- **Ensemble stacking** → Combine multiple models for better performance  
- **Deployment** → Flask or Streamlit web interface  
- **Reproducibility** → Dockerized version of the project  

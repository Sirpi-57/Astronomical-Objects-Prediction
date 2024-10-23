# **Galaxy, Star, or Quasar Classification Project**

## **Project Overview**

This project aims to classify celestial objects (Galaxy, Star, or Quasar) using data from the Apache Point Observatory. The classification is performed on a dataset containing various features like redshift, magnitude values in different filters, and other observational data. The primary goal is to build a machine learning model that can accurately predict whether a celestial object is a **Galaxy**, **Star**, or **Quasar** (QSO) based on the given features.

### **Evaluation Metric**

The performance of the models is measured using **accuracy**—the percentage of correctly classified objects. The project evaluates multiple models, including logistic regression, K-Nearest Neighbors (KNN), Random Forest, and XGBoost, with accuracy and training time being the primary factors for model selection.

---

## **Dataset**

The dataset contains 10,000 records with the following columns:

| Column Name | Description |
| ----- | ----- |
| objid | Object ID |
| ra | Right Ascension (celestial coordinate) |
| dec | Declination (celestial coordinate) |
| u, g, r, i, z | Magnitudes in different photometric bands |
| run, rerun, camcol, field | Observation-specific metadata |
| specobjid | Spectroscopic object ID |
| class | The target variable (Star, Galaxy, Quasar) |
| redshift | Object's redshift value |
| plate | Plate number |
| mjd | Modified Julian Date |
| fiberid | Fiber ID used in spectroscopic observation |

---

## **Project Workflow**

### **1\. Data Preprocessing**

**Encoding Target Labels:**  
The `class` column (Galaxy, Star, Quasar) is encoded into numerical values using a mapping:

`mapp = {'STAR': 0, 'GALAXY': 1, 'QSO': 2}`  
`df_encode['class'] = df_encode['class'].map(mapp)`

**Principal Component Analysis (PCA):**  
PCA is applied on the magnitude columns (`u, g, r, i, z`) to reduce dimensionality. Only the top 3 components are kept:

`pca = PCA(n_components=3)`  
`ugriz = pca.fit_transform(df_encode[['u', 'g', 'r', 'i', 'z']])`

**Feature Scaling:**  
The features are scaled using **MinMax Scaler** to bring them into the same range for better performance in distance-based models:

`scaler = MinMaxScaler()`  
`scaled_features = scaler.fit_transform(df_encode)`

### **2\. Model Training & Evaluation**

Several machine learning models were trained and evaluated on the dataset:

**Logistic Regression:**

`logreg.fit(X_train, y_train)`  
`accuracy_logreg = logreg.score(X_test, y_test)`  
**K-Nearest Neighbors (KNN):**

`neigh.fit(X_train, y_train)`  
`accuracy_neigh = neigh.score(X_test, y_test)`

**Random Forest:**

`rfc.fit(X_train, y_train)`  
`accuracy_rfc = rfc.score(X_test, y_test)`

**XGBoost:**

`xgb.fit(X_train, y_train)`  
`accuracy_xgb = xgb.score(X_test, y_test)`

**Support Vector Machine (SVM):**

`clf.fit(X_train, y_train)`  
`accuracy_clf = clf.score(X_test, y_test)`

### **3\. Evaluation Results:**

After training, the models were evaluated using the test set, and their performance was compared based on accuracy and training time.

| Model | Accuracy | Training Time (seconds) |
| ----- | ----- | ----- |
| **XGBoost** |     99.36% |            0.3551 |
| **Random Forest** |     99.26% |            1.6528 |
| **Naive Bayes** |     98.00% |            0.0019 |
| **SVM** |     94.50% |            0.6431 |
| **K-Nearest Neighbors** |     93.63% |            0.2516 |
| **Logistic Regression** |     91.57% |            0.3624 |

From these results, **XGBoost** was selected as the best model for its balance of high accuracy and reasonable training time.

### 

### **4\. Cross-Validation**

Cross-validation was used to evaluate model performance across different splits of the data. The mean accuracy and standard deviation were calculated for each model to ensure robustness.

Example:

`cross_val_xgb = cross_val_score(xgb, X, y, cv=10)`  
`mean_accuracy = cross_val_xgb.mean()`  
`std_accuracy = cross_val_xgb.std()`

### **5\. Feature Importance (for XGBoost):**

The most important features identified by the XGBoost model were:

* **redshift** (dominantly important)  
* PCA components derived from `u, g, r, i, z`  
* Other features like `plate`, `fiberid`, `mjd`, etc.

`importances = pd.DataFrame({'Feature': df_encode.drop('class', axis=1).columns, 'Importance': xgb.feature_importances_})`  
`importances = importances.sort_values(by='Importance', ascending=False)`

---

## **Conclusion**

The project demonstrates that XGBoost is the best-performing model for this classification task, achieving high accuracy (99.36%) while maintaining reasonable training time. The results indicate that **redshift** is the most critical feature in determining the object class (Galaxy, Star, or Quasar), with the PCA-transformed photometric magnitudes also contributing to the model's predictive power.


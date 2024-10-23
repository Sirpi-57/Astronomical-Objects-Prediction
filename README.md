**Classification of Galaxy, Star, or Quasar**

**Objective:**
This project aims to help astronomers at the Apache Point Observatory predict the type of celestial object—whether it is a Galaxy, Star, or Quasar—based on data recorded by the optical telescope. The classification is done using a machine learning model, and the evaluation metric used is accuracy. The project involves cleaning the dataset, performing exploratory data analysis (EDA), applying feature engineering, and testing various machine learning algorithms.

**Explanations of Concepts:**

**1. Training a Model:**

In this project, we tested several machine learning models (Logistic Regression, K-Nearest Neighbors, Random Forest, XGBoost, etc.) and fine-tuned them based on their performance. Fine-tuning involves adjusting the model's hyperparameters to achieve the best balance of accuracy and training time.

**2. Principal Component Analysis (PCA):**

PCA was applied to reduce the dimensionality of the dataset while retaining important variance. We focused on the features ‘u’, ‘g’, ‘r’, ‘i’, and ‘z’, transforming them into three principal components. This helped simplify the model without losing crucial information.

**3. Model Selection:**
   
Several models were trained and evaluated on accuracy and training time. The models tested include:

**Logistic Regression
K-Nearest Neighbors
Naive Bayes
Random Forest
XGBoost
Support Vector Machine (SVM)**

After evaluating the models, XGBoost was chosen as the best performing model, balancing accuracy and speed.

**4. Cross Validation:**

Cross-validation was performed to assess model performance more reliably by splitting the data into training and testing sets multiple times. This helped ensure the model’s generalizability to unseen data.

**5. Feature Importance:**

Feature importance from the XGBoost model was calculated, identifying redshift as the most crucial feature for classifying celestial objects, followed by the principal components derived from PCA.

**6. Handling Imbalanced Data:**

Despite some class imbalance (fewer quasars than stars and galaxies), the models performed well, achieving high precision and recall for all classes.

**Project Deliverables:**

**Data Cleaning and Preprocessing:**

Handled missing values, outliers, and skewness.
Scaled the features using MinMax Scaler to normalize data for better model performance.

**Exploratory Data Analysis (EDA):**

Distribution of features analyzed.
Correlation analysis between features and the target class.

**Feature Engineering:**

Applied PCA to reduce feature dimensionality.
Mapped the 'class' column to numerical values for model compatibility ({'STAR': 0, 'GALAXY': 1, 'QSO': 2}).

**Model Development and Testing:**

Logistic Regression, K-NN, Random Forest, XGBoost, Naive Bayes, and SVM models were trained.
Models were evaluated using accuracy, precision, recall, and f1-score on the test dataset.

**Model Selection:**

**XGBoost** was selected as the best model based on its accuracy and cross-validation performance.

**Feature Importance:**

Provided insights into the most significant features impacting classification.

**Future Scope:**

**Improved Feature Engineering:**

Explore additional feature extraction methods, such as creating new features based on astronomical data characteristics.

**Integration with Real-time Data:**

Connect this model to a real-time data stream from the Apache Point Observatory for continuous classification.

**Deployment as a Web Application:**

Deploy the model in a user-friendly interface for astronomers to easily input new data and receive predictions.

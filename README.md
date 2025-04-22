# SENTIMENT-ANALYSIS
*COMPANY* : CODTECH IT SOLUTIONS
*NAME* : RAMJI MISHRA
*INTERN ID* : CT04WT148
*DOMAIN* : DATA ANALYTICS
*DURATION* : 4 WEEKS
*MENTOR* : NEELA SANTHOSH
## # **Twitter Sentiment Analysis Using Count Vectorizer and TF-IDF with Machine Learning Algorithms**

## **Introduction**
Twitter sentiment analysis is a natural language processing (NLP) task that involves classifying tweets into positive, negative, or neutral sentiments based on their textual content. This project explores the effectiveness of different machine learning algorithms—**Decision Tree, Logistic Regression, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN)**—in predicting sentiment using two popular text representation techniques: **Count Vectorizer and TF-IDF (Term Frequency-Inverse Document Frequency)**.

The project also incorporates **Grid Search Cross-Validation (GridSearchCV)** for hyperparameter tuning to optimize model performance. The goal is to compare the efficiency of these algorithms in sentiment classification and determine which combination of text representation and model yields the best results.

---

## **Methodology**

### **1. Data Collection and Preprocessing**
The dataset consists of labeled Twitter tweets categorized into positive, negative, or neutral sentiments. The preprocessing steps include:
- **Text Cleaning:** Removing URLs, special characters, and stopwords.
- **Tokenization:** Breaking text into individual words or tokens.
- **Lemmatization/Stemming:** Reducing words to their base forms (e.g., "running" → "run").

### **2. Feature Extraction**
Two primary techniques are used to convert text into numerical features:
1. **Count Vectorizer:**  
   - Converts text into a matrix of token counts.
   - Represents each document as a vector of word frequencies.
2. **TF-IDF (Term Frequency-Inverse Document Frequency):**  
   - Weighs words based on their importance in a document relative to their frequency across all documents.
   - Helps reduce the impact of overly common words (e.g., "the", "and").

### **3. Machine Learning Models**
The following algorithms are implemented and evaluated:
1. **Decision Tree:**  
   - A tree-based model that splits data based on feature importance.
   - Prone to overfitting but interpretable.
2. **Logistic Regression:**  
   - A linear model for binary/multi-class classification.
   - Works well with high-dimensional sparse data (like text).
3. **Support Vector Machine (SVM):**  
   - Finds the optimal hyperplane to separate classes.
   - Effective in high-dimensional spaces.
4. **K-Nearest Neighbors (KNN):**  
   - Classifies based on the majority vote of nearest neighbors.
   - Sensitive to feature scaling.

### **4. Hyperparameter Tuning with GridSearchCV**
- **GridSearchCV** performs an exhaustive search over specified parameter values to find the best model configuration.
- Example parameters tuned:
  - **Logistic Regression:** `C` (regularization strength), `penalty` (L1/L2).
  - **SVM:** `C`, `kernel` (linear, RBF).
  - **Decision Tree:** `max_depth`, `min_samples_split`.
  - **KNN:** `n_neighbors`, `weights` (uniform, distance).

### **5. Model Evaluation**
Performance is measured using:
- **Accuracy:** Overall correctness of predictions.
- **Precision, Recall, F1-Score:** For imbalanced datasets.
- **Confusion Matrix:** Visualizes true vs. predicted labels.

---

## **Results and Discussion**
- **Count Vectorizer vs. TF-IDF:**  
  - TF-IDF often performs better by reducing bias towards frequent words.
  - Count Vectorizer may capture more raw frequency patterns but can be noisy.
  
- **Model Comparisons:**  
  - **Logistic Regression & SVM** generally perform well due to their ability to handle high-dimensional text data.
  - **Decision Trees** may overfit unless pruned properly.
  - **KNN** can be computationally expensive and less effective without proper scaling.

- **Impact of GridSearchCV:**  
  - Improves model accuracy by optimizing hyperparameters.
  - Helps prevent overfitting in models like Decision Trees and SVM.

---

## **Conclusion**
This project demonstrates that **TF-IDF combined with Logistic Regression or SVM** yields the highest accuracy in Twitter sentiment analysis. **GridSearchCV** significantly enhances model performance by fine-tuning parameters. While **Count Vectorizer** is simpler, **TF-IDF** provides better discriminative power for sentiment classification. Future work could explore deep learning models (e.g., LSTM, BERT) for improved contextual understanding.

---

## **Key Takeaways**
- **Text representation (Count Vectorizer vs. TF-IDF)** plays a crucial role in model performance.
- **Logistic Regression and SVM** are highly effective for text classification.
- **Hyperparameter tuning (GridSearchCV)** is essential for optimizing machine learning models.
- **Decision Trees and KNN** may require additional preprocessing and tuning for competitive performance.

This project provides a comprehensive framework for sentiment analysis on Twitter data, serving as a foundation for further NLP and machine learning research.


# Fake News Detection & Misinformation Tracker  

Detecting fake and misleading news articles using **Natural Language Processing (NLP)** and **Machine Learning** techniques.  
This project compares multiple models (Logistic Regression, Random Forest, XGBoost, etc.) and achieves the best performance using **XGBoost with TF-IDF features**.  

---

##  Objective  
Build a robust machine learning system to:  
- Identify whether a news article is **Fake** or **Real**  
- Compare traditional ML algorithms on text classification tasks  
- Understand which models perform better on high-dimensional sparse features  

---

##  Dataset  
- Source: [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news)  
- Contains thousands of labeled news articles with fields like **title, text, subject, date**  
- Target variable:  
  - **1 → Fake News**  
  - **0 → Real News**  

---

##  Steps Followed  

1. **Exploratory Data Analysis (EDA)**  
   - Word distributions, text length analysis, most frequent words  
   - WordClouds for Fake vs Real news  

2. **Preprocessing**  
   - Tokenization, lowercasing, removing stopwords, punctuation, lemmatization  
   - Feature extraction using **TF-IDF Vectorizer**  

3. **Model Training**  
   - Logistic Regression  
   - Naive Bayes  
   - Decision Tree  
   - Random Forest  
   - KNN  
   - XGBoost  

4. **Evaluation Metrics**  
   - Accuracy, Precision, Recall, F1-Score  
   - Confusion Matrix  

---

##  Results  

| Model              | Accuracy | F1-Score |
|--------------------|----------|----------|
| Logistic Regression | 90%     | 0.89     |
| Naive Bayes        | 88%     | 0.87     |
| Decision Tree      | 85%     | 0.84     |
| Random Forest      | 92%     | 0.91     |
| KNN                | 82%     | 0.80     |
| **XGBoost**        | **95%** | **0.94** |

 **XGBoost performed the best** due to its ability to handle **high-dimensional sparse TF-IDF features** efficiently.  

---

##  Key Insights  

- **TF-IDF over Bag of Words:** TF-IDF reduces the weight of common words ("the", "is") and gives importance to informative words, leading to better performance.  
- **Why XGBoost works better:** Tree-based ensemble models like Random Forest and XGBoost handle sparse, high-dimensional data better than models like KNN or single Decision Trees.  
- **Logistic Regression & Naive Bayes:** Good baseline models for text classification, but struggle when relationships are more complex.  

---

##  Next Steps  

- Deploy the model as a **Streamlit web app** or Flask API  
- Improve robustness with **transformer-based models (BERT, DistilBERT)**  
- Handle **class imbalance** using oversampling techniques (SMOTE)  
- Add **explainability** with SHAP/LIME for model transparency  

---

##  How to Run  

1. Clone this repository:  
   ```bash
   git clone https://github.com/hasINI-123-006/Fake-News-Detection-Using-Machine-Learning.git
   cd Fake-News-Detection-Using-Machine-Learning


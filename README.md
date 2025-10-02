
# Fake News Detection & Misinformation Tracker  

Detecting fake and misleading news articles using **Natural Language Processing (NLP)** and **Machine Learning** techniques.  
This project compares multiple models (Logistic Regression, Random Forest, XGBoost, etc.) and achieves the best performance using **XGBoost with TF-IDF features**.  

---

##  Objective  
The goal of this project is to build a machine learning pipeline that can classify news articles as **Fake** or **Real**. Specifically, it focuses on:
- Exploring and preprocessing text data using techniques like tokenization, lemmatization, and TF-IDF vectorization  
- Comparing the performance of multiple traditional machine learning models, including Logistic Regression, Naive Bayes, Random Forest, Decision Tree, KNN, and
  XGBoost  
- Understanding which models perform better on high-dimensional sparse text features
 
---

##  Dataset  
This project utilizes the **Fake and Real News Dataset** by Clément Bisaillon, available on Kaggle:
[Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
The dataset includes:
- `True.csv`: Contains real news articles.
- `Fake.csv`: Contains fake news articles.
These files are used for training and evaluating the machine learning models in this project.
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
| Logistic Regression | 98%     | 0.98     |
| Naive Bayes        | 90%     | 0.90     |
| Decision Tree      | 99%     | 0.99     |
| Random Forest      | 100%     | 1.00     |
| KNN                | 92%     | 0.92     |
| **XGBoost**        | **100%** | **1.00** |

** Why XGBoost Performed Best**
XGBoost outperformed other models because it builds trees sequentially, correcting errors from previous trees, handles sparse TF-IDF features efficiently, and includes regularization to prevent overfitting. These strengths make it highly effective for text classification tasks like fake news detection.

---

##  Key Insights  

- **TF-IDF over Bag of Words:** TF-IDF reduces the weight of common words ("the", "is") and gives importance to informative words, leading to better performance.  
- **Why XGBoost works better:** Tree-based ensemble models like Random Forest and XGBoost handle sparse, high-dimensional data better than models like KNN or single Decision Trees.  
- **Logistic Regression & Naive Bayes:** Good baseline models for text classification, but struggle when relationships are more complex.  

---

##  Future Enhancements  

This project currently uses traditional machine learning models with TF-IDF features to classify news as fake or real. In the future, it could be extended by:

- Deploying the model as a simple web app using **Streamlit** or **Flask**  
- Experimenting with transformer-based models like **BERT** for potentially better accuracy  
- Adding explainability tools like **SHAP** or **LIME** to understand why the model makes certain predictions

---

##  How to Run  

1. Clone this repository:  
   ```bash
   git clone https://github.com/hasINI-123-006/Fake-News-Detection-Using-Machine-Learning.git
   cd Fake-News-Detection-Using-Machine-Learning


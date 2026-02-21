
# Fake News Detection & Misinformation Tracker  

Detecting fake and misleading news articles using **Natural Language Processing (NLP)** and **Machine Learning** techniques.  

---

##  Objective  
The goal of this project is to build a machine learning pipeline that can classify news articles as **Fake** or **Real**. Specifically, it focuses on:
- Exploring and preprocessing text data using techniques like tokenization, lemmatization, and TF-IDF vectorization  
- Comparing the performance of multiple traditional machine learning models, including Logistic Regression, Naive Bayes, Random Forest, Decision Tree, KNN, and XGBoost  
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
  - **0 → Fake News**  
  - **1 → Real News**  

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
| Naive Bayes        | 93%     | 0.93     |
| Decision Tree      | 99%     | 0.99     |
| Random Forest      | 100%     | 0.99     |
| KNN                | 67%     | 0.58     |
| XGBoost            | 100%     | 0.99    |

---

##  Key Insights  

- **TF-IDF over Bag of Words:** TF-IDF reduces the weight of common words ("the", "is") and gives importance to informative words, leading to better performance.  
- **Overall Model Performance:** Most models performed very well on this dataset. Tree-based models like Random Forest and XGBoost achieved extremely high accuracy, while Logistic Regression also delivered strong and consistent results. Naive Bayes worked well as a reliable baseline model, whereas KNN was comparatively less effective for this task.  
- **Why Accuracy is so high:** The dataset contains clear stylistic differences between fake and real news articles. Real news tends to follow a formal journalistic tone, while fake news often uses different wording, structure, and vocabulary. Because TF-IDF captures these distinguishing words effectively, the models are able to separate the two classes with high confidence, leading to very high accuracy scores.

---

## Deployment 

The project includes an interactive web application built using Streamlit.
The application allows users to:
  - Enter any news article text
  - Click the Predict button
  - Instantly receive:
       - The predicted label (Fake or Real)
       - The model’s confidence score
The app runs locally and automatically opens in the browser.
**How to run** 

1. Clone this repository:  
   ```bash
   git clone https://github.com/hasINI-123-006/Fake-News-Detection-using-Machine-Learning.git
   cd Fake-News-Detection-Using-Machine-Learning
2. Install required dependencies 
   pip install -r ML_pj_requirements.txt
3. Run the application
   python -m streamlit run MLpj_app.py

---

### Limitations

- The dataset contains noticeable stylistic differences between fake and real news articles, which may contribute to very high classification accuracy.
- The model is trained on a specific dataset and may require retraining to generalize well to different domains or evolving writing styles.
- The current deployment runs locally and is not yet publicly hosted.

---

##  Future Enhancements  
 
- Experimenting with transformer-based models like **BERT** for potentially better accuracy  
- Adding explainability tools like **SHAP** or **LIME** to understand why the model makes certain predictions

---



# Multilingual Mobile App Review Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Library-Scikit--learn-orange)
![Jupyter Notebook](https://img.shields.io/badge/Notebook-Jupyter-orange)
![Status](https://img.shields.io/badge/Status-Analysis%20Complete-success)

## 📖 Project Overview

This project performs sentiment analysis on a dataset of multilingual mobile app reviews. The goal is to classify user reviews into **Positive** or **Negative** sentiments based on the text content and a rating threshold. The project involves comprehensive data cleaning, preprocessing, and a comparative evaluation of multiple machine learning models.

**Key Insight:** The analysis revealed a significant class imbalance and that standard TF-IDF with classical ML models struggles to effectively identify positive sentiments in this multilingual context. This serves as a strong baseline and highlights areas for future improvement.

## 📊 Dataset

The dataset `multilingual_mobile_app_reviews_2025.csv` contains 2,514 mobile app reviews with 15 features.

**Main Features:**
*   `review_text`: The main content of the user review (primary feature for analysis).
*   `rating`: Numerical rating from 1 to 5.
*   `review_language`: Language code of the review (e.g., 'es', 'ru', 'no').
*   Additional metadata: `user_country`, `user_age`, `app_name`, `app_category`, `device_type`, etc.

**Label Creation:**
Reviews were binarized for sentiment classification:
*   **Positive (1)**: `rating >= 3.5`
*   **Negative (0)**: `rating < 3.5`

**Class Distribution:**
*   Negative Reviews: 1482
*   Positive Reviews: 936

## 🛠️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/multilingual-review-sentiment-analysis.git
    cd multilingual-review-sentiment-analysis
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *The `requirements.txt` file should include:*
    ```txt
    pandas==1.5.3
    numpy==1.23.5
    scikit-learn==1.2.2
    matplotlib==3.7.0
    seaborn==0.12.2
    jupyter==1.0.0
    ```

## 🚀 Usage

1.  Place the `multilingual_mobile_app_reviews_2025.csv` file in the project directory.
2.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook Multilingual_Review_Sentiment_Analysis.ipynb
    ```
3.  Run the notebook cells sequentially to:
    *   Load and explore the data.
    *   Clean and preprocess the reviews.
    *   Train and evaluate multiple machine learning models.
    *   Generate performance metrics and visualizations.

## 🔬 Methodology

### 1. Data Preprocessing
*   **Handling Missing Values:**
    *   Text reviews: Filled with `"unknown"`.
    *   Numerical ratings: Imputed with the median value.
    *   Categorical variables (`user_country`, `user_gender`, `app_version`): Imputed with the mode.
*   **Text Vectorization:** Used `TfidfVectorizer` with English stop words removed, limited to the top 5000 features.

### 2. Models Trained & Compared
We compared the performance of four classic machine learning algorithms:
1.  **Logistic Regression**
2.  **Multinomial Naive Bayes**
3.  **Random Forest Classifier**
4.  **Support Vector Machine (LinearSVC)**

### 3. Evaluation Metrics
Models were evaluated on a stratified 80/20 train-test split using:
*   Accuracy
*   Precision
*   Recall
*   F1-Score
*   Confusion Matrix

## 📈 Results and Visualization

### Model Performance Summary
| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 61.0% | 30.0% | 1.6% | 0.030 |
| **Naive Bayes** | 59.4% | 30.0% | 4.7% | 0.081 |
| **Random Forest** | 61.0% | 37.5% | 3.1% | 0.058 |
| **SVM** | **59.8%** | **34.4%** | **5.7%** | **0.098** |

**Conclusion:** While accuracy is around 60-61%, all models fail to identify positive reviews effectively (extremely low recall). The SVM model slightly outperforms others on F1-Score but performance is still poor.

### Visualizations

**1. Rating Distribution**
A histogram showing the spread of numerical ratings in the dataset.
![Rating Distribution](images/rating_distribution.png)

**2. Class Distribution Bar Plot**
A countplot showing the imbalance between negative (0) and positive (1) labels.
![Class Distribution](images/class_distribution.png)

**3. Confusion Matrix Heatmap (e.g., Naive Bayes)**
A heatmap visualizing the True/False Positives/Negatives, clearly showing the model's bias towards the majority class.
![Confusion Matrix](images/confusion_matrix.png)

## ‼️ Limitations and Future Work

**Limitations:**
*   **Class Imbalance:** The dataset is skewed towards negative reviews.
*   **Multilingual Text:** Using TF-IDV with English stopwords is suboptimal for non-English text.
*   **Feature Set:** Only the text was used, ignoring potentially useful metadata.

**Future Work:**
*   **Address Imbalance:** Use techniques like SMOTE, class weights, or different evaluation metrics (Precision-Recall AUC).
*   **Advanced NLP:** Implement language detection and use multilingual embeddings (e.g., `BERT`, `fastText`, `SentenceTransformers`).
*   **Feature Engineering:** Incorporate metadata (e.g., `app_category`, `user_country`) into the model.
*   **Modeling:** Experiment with deep learning models (e.g., RNNs, Transformers) and hyperparameter tuning.

## 👥 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/your-username/multilingual-review-sentiment-analysis/issues).

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

## 🙏 Acknowledgments

*   Dataset provided for analytical purposes.
*   Built with the incredible Python data science ecosystem: Pandas, Scikit-learn, Matplotlib, and Seaborn.

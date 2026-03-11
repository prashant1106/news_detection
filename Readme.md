# Fake News Detector

**Author:** Prashant Kumar Jha

**Date:** 11-03-2026

Fake News Detector is a machine learning project that detects whether a news article is **Fake or Real** using a **Multinomial Naïve Bayes classifier implemented from scratch in Python**.

The project performs **text preprocessing, feature extraction, model training, evaluation, and prediction**, and allows users to classify their own news articles as real or fake.

---

# Overview

The spread of **fake news** has become a major challenge in the digital era. This project aims to build a **Fake News Detection System** that automatically detects if a  news article is **Real or Fake**.

The model uses **Natural Language Processing (NLP)** techniques and a **custom implementation of Multinomial Naïve Bayes** for text classification.

The pipeline includes:

* Data preprocessing
* Text cleaning
* Feature extraction using **Bag of Words**
* Training a **Naïve Bayes classifier**
* Model evaluation
* Visualization of results
* Real-time user input prediction

---

# Features

* Loads and merges **Fake and Real news datasets**
* Performs **text preprocessing and cleaning**
* Removes **stopwords and applies stemming**
* Generates a **Term-Document Matrix**
* Implements **Multinomial Naïve Bayes from scratch**
* Evaluates model using **Accuracy and Confusion Matrix**
* Visualizes results using **Seaborn heatmap**
* Allows **user to input their news and check it's authenticity**

---

# Dataset Information

The dataset used in this project contains labeled **Fake and True news articles**.

**Dataset Source:**
https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/

### Files Used

| File       | Description                 |
| ---------- | --------------------------- |
| `Fake.csv` | Contains fake news articles |
| `True.csv` | Contains real news articles |

### Labels

| Label | Meaning   |
| ----- | --------- |
| `0`   | Fake News |
| `1`   | Real News |

---

# Technologies Used

| Technology              | Purpose                          |
| --------------------    | -------------------------------- |
| **Python**              | Programming language             |
| **Pandas**              | Data manipulation                |
| **NumPy**               | Numerical operations             |
| **NLTK**                | Text preprocessing               |
| **Scikit-learn**        | Data splitting and vectorization |
| **Matplotlib**          | Visualization                    |
| **Seaborn**             | Confusion matrix heatmap         |
| **Jupyter Notebook**    | Development environment          |
| **Visuacl Studio Code** | Code Editor                      |

---

# Project Workflow

## 1️. Data Loading

Load the datasets using **Pandas**.

```python
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")
```

---

## 2. Data Labeling

Assign labels to the datasets:

* Fake News → `0`
* True News → `1`

```python
fake["label"] = 0
true["label"] = 1
```

---

## 3️. Dataset Merging

Merge the datasets and shuffle the data.

```python
data = pd.concat([fake, true], axis=0)
data = data.sample(frac=1).reset_index(drop=True)
```

---

## 4️. Feature Engineering

Combine **title** and **text** into a single column.

```python
data["content"] = data["title"] + " " + data["text"]
```

---

## 5️. Text Preprocessing

Text preprocessing includes:

* Converting text to lowercase
* Removing non-alphabet characters using regex
* Removing stopwords
* Applying Porter stemming


```python
def clean_text(text):

    text = text.lower()

    text = re.sub('[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [ps.stem(word) for word in words if word not in stop_words]

    return " ".join(words)
```

---

## 6️. Text Vectorization

Convert text data into **Document-Term Matrix** using **Bag of Words**.

```python
vectorizer = CountVectorizer(max_features=5000)
x = vectorizer.fit_transform(corpus).toarray()
```

---

## 7️. Train-Test Split

Split the dataset into training and testing sets.

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
```

---

## 8️. Model Implementation

The **Multinomial Naïve Bayes classifier is implemented from scratch**.

The model calculates:

* Prior probabilities
* Word frequencies per class
* Conditional probabilities using **Laplace smoothing**

Prediction formula:

```
log(P(c)) + Σ xi * log(P(wi | c))
```

---

## 9️. Model Evaluation

The model is evaluated using:

* **Accuracy Score**
* **Confusion Matrix**

```python
accuracy_score(y_test, y_predict)
confusion_matrix(y_test, y_predict)
```

---

# Installation Instructions

Clone the repository:

```bash
git clone https://github.com/yourusername/news_detection.git
```

Navigate to the project directory:

```bash
cd news_detection
```

Install required dependencies:

```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn
```

Download NLTK stopwords:

```python
import nltk
nltk.download('stopwords')
```

---

# How to Run the Project

1. Open the notebook:

```
news_detection.ipynb
```

2. Run all cells in the notebook.

3. Enter a news article when prompted.

Example:

```
Enter the news article:
Scientists discovered a new renewable energy source that could replace fossil fuels.
```

---

# Output

```
Accuracy: 0.9472605790645879
```

User prediction example:

```
Enter the news article:
Breaking: Government secretly replaced moon with a giant LED screen.

It is FAKE news.
```

---

# Confusion Matrix Visualization

The confusion matrix is visualized using **Seaborn heatmap**.

```python
sns.heatmap(cm, annot=True, fmt="d", cmap="plasma")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Fake News Detection")
plt.show()
```

Confusion matrix:

|                 | Predicted Fake | Predicted Real |
| --------------- | -------------- | -------------- |
| **Actual Fake** | 5543           | 267            |
| **Actual Real** | 325            | 5090           |

This visualization helps analyze:

* False Positives
* False Negatives
* Model accuracy

---

# Future Improvements

Possible improvements for the project:

* Use **TF-IDF vectorization**
* Train models like **Logistic Regression, SVM**
* Implement **Deep Learning models (LSTM / Transformers)**
* Deploy the model using **Flask or FastAPI**
* Build a **web interface**
* Integrate **real-time news scraping**
* Improve preprocessing for **numbers and named entities**

---

# Author

**Prashant Kumar Jha**

GitHub:
https://github.com/prashant1106

---

# License

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute this project for educational and research purposes.

---

If you found this project useful, consider **starring the repository**.

Thank you for reading this and giving love to this project.

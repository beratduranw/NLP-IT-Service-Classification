# README

## Overview
This project involves processing and analyzing a dataset of support ticket documents. The main tasks include removing stopwords, generating word clouds, and analyzing text data using TF-IDF vectorization. The dataset is categorized into various topic groups.

## Requirements
To run the code, you need the following libraries:
- pandas
- matplotlib
- seaborn
- numpy
- nltk
- wordcloud
- scikit-learn

Make sure to install these packages using pip if you haven't already:
```sh
pip install pandas matplotlib seaborn numpy nltk wordcloud scikit-learn
```

## Dataset
The dataset is stored in a CSV file named `all_tickets_processed_improved_v3.csv`. It contains the following columns:
- `Document`: The text of the support ticket.
- `Topic_group`: The category or topic group of the ticket.

## Code Explanation

### 1. Import Libraries and Setup
The initial code imports necessary libraries and sets up configurations:
```python
from warnings import filterwarnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

stop_words = stopwords.words('english')

filterwarnings('ignore')
```

### 2. Read Data
Load the dataset and display the first few rows:
```python
df = pd.read_csv("all_tickets_processed_improved_v3.csv")
df.head()
```

### 3. Remove Stopwords
Define a function to remove stopwords from the documents:
```python
def remove_stopwords(text):
    words = text.lower().split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

df['Document'] = df['Document'].apply(remove_stopwords)
```

### 4. Create Word Cloud
Create a word cloud to visualize the most frequent words in the documents:
```python
def create_cloud(data_frame):
    text = data_frame["Document"].str.cat(sep=" ").lower()
    wordcloud = WordCloud(max_font_size=40).generate(text)

    plt.figure(figsize=(8, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Most Frequent Words")
    plt.show()

#create_cloud(df)
```

### 5. Remove Specific Words
Remove specific words that are commonly found in the dataset but do not add value to the analysis:
```python
words_to_remove = ["ga", "kind", "hello", "please", "let", "help", "best", "regards", "icon", "dear", "per", "hi", "thanks", "thank", "importance", "high", "issue", "ab", "abc"]

def remove_words(text):
    return " ".join([word for word in text.lower().split() if word not in words_to_remove])

df["Document"] = df["Document"].apply(remove_words)
#create_cloud(df)
```

### 6. See Example Row
Define a function to display an example row from the dataset:
```python
def see_example_row(index: int):
    first_document = df["Document"].iloc[index]
    words = first_document.split()[:100]
    print(" ".join(words))
    print(df["Topic_group"].iloc[index])

see_example_row(1)
```

### 7. TF-IDF Vectorization
Perform TF-IDF vectorization on the documents:
```python
vectorizer = TfidfVectorizer(stop_words='english')
vectorizer.fit(df["Document"])
```

Transform the documents into a TF-IDF matrix:
```python
tfidf_matrix = vectorizer.transform(df["Document"])
```

### 8. Group Documents by Topic
Group the documents by their topic group:
```python
grouped_documents = {}
for doc, topic in zip(df["Document"], df["Topic_group"]):
    if topic not in grouped_documents:
        grouped_documents[topic] = []
    grouped_documents[topic].append(doc)
grouped_documents
```

### 9. Analyze TF-IDF for Each Topic Group
Analyze the TF-IDF scores for each topic group:
```python
for topic, group_documents in grouped_documents.items():
    print(f"\n** Topic Group: {topic} **")
    print((group_documents[0]))

    group_vectorizer = TfidfVectorizer(stop_words='english')
    tfid_matrix = group_vectorizer.fit_transform(group_documents)
    feature_names = group_vectorizer.get_feature_names_out()

    overall_tfidf_sum = tfidf_matrix.sum(axis=0)
    overall_top_features = sorted(zip(feature_names, overall_tfidf_sum), key=lambda x: x[1], reverse=True)[:10]
    #print(overall_top_features)
```

## Usage
To run the analysis, execute the code cells in sequence. Make sure the dataset file is in the same directory as the notebook. You can modify the functions or add new ones to further explore the data.

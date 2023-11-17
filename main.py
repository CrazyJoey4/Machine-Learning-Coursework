import pandas as pd
import re
import nltk
nltk.download('wordnet')
nltk.download('punkt')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from gensim.models import Word2Vec

# For Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# For SVM Classifier
from sklearn.svm import SVC

#For Optimize (Stacking Ensemble)
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression




# Load the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Drop unnecessary columns
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

# Rename columns
df = df.rename(columns={"v1": "label", "v2": "text"})

# Map labels to binary values
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Text preprocessing functions
def preprocess_text(text):
    # Remove unnecessary characters and convert to lowercase
    text = re.sub("[^a-zA-Z]", " ", text.lower())
    # Tokenization
    words = text.split()
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words("english")]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    # Join the words back into a string
    return ' '.join(words)

# Apply preprocessing to the entire dataset
df['text'] = df['text'].apply(preprocess_text)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Convert text data to numerical features using CountVectorizer
count_vectorizer = CountVectorizer()
X_train_counts = count_vectorizer.fit_transform(X_train)
X_test_counts = count_vectorizer.transform(X_test)

# Convert term frequency to TF-IDF
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)


# Tokenize the text into words
tokenized_text = df['text'].apply(word_tokenize)

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=1, workers=4)

# Function to get the vector representation of a sentence
def get_vector(sentence, model):
    vector = [model.wv[word] for word in sentence if word in model.wv]
    return sum(vector) / len(vector) if vector else [0] * model.vector_size

# Apply Word2Vec to the entire dataset
df['text_vector'] = tokenized_text.apply(lambda x: get_vector(x, word2vec_model))

# Split the dataset into training and testing sets
X_train_w2v, X_test_w2v, y_train_w2v, y_test_w2v = train_test_split(df['text_vector'], df['label'], test_size=0.2, random_state=42)


# First algorithm : Naive Bayes Classifier
# Create Naive Bayes classifier
naive_bayes_classifier = MultinomialNB()

# Train the classifier on the TF-IDF transformed data
naive_bayes_classifier.fit(X_train_tfidf, y_train)

# Predictions on the test set
predictions_nb = naive_bayes_classifier.predict(X_test_tfidf)

# Evaluate the performance
accuracy_nb = accuracy_score(y_test, predictions_nb)
classification_report_nb = classification_report(y_test, predictions_nb)

print("Naive Bayes Classifier:")
print("Accuracy:", accuracy_nb)
print("Classification Report:\n", classification_report_nb)



# Second algorithm : Support Vector Machines (SVM) Classifier
# Create SVM classifier
svm_classifier = SVC(kernel='linear')  # You can experiment with different kernels (linear, rbf, etc.)

# Train the classifier on the TF-IDF transformed data
svm_classifier.fit(X_train_tfidf, y_train)

# Predictions on the test set
predictions_svm = svm_classifier.predict(X_test_tfidf)

# Evaluate the performance
accuracy_svm = accuracy_score(y_test, predictions_svm)
classification_report_svm = classification_report(y_test, predictions_svm)

print("\nSupport Vector Machines (SVM) Classifier:")
print("Accuracy:", accuracy_svm)
print("Classification Report:\n", classification_report_svm)


# To optimize the models : Stacking Ensemble
# Define base classifiers
base_classifiers = [
    ('decision_tree', DecisionTreeClassifier()),
    ('random_forest', RandomForestClassifier()),
    ('naive_bayes', MultinomialNB()),
    ('svm', SVC(kernel='linear'))
]

# Create the stacking classifier
stacking_classifier = StackingClassifier(estimators=base_classifiers, final_estimator=LogisticRegression())

# Train the stacking classifier on the TF-IDF transformed data
stacking_classifier.fit(X_train_tfidf, y_train)

# Predictions on the test set
predictions_stacking = stacking_classifier.predict(X_test_tfidf)

# Evaluate the performance of the stacking classifier
accuracy_stacking = accuracy_score(y_test, predictions_stacking)
classification_report_stacking = classification_report(y_test, predictions_stacking)

print("\nStacking Ensemble Classifier:")
print("Accuracy:", accuracy_stacking)
print("Classification Report:\n", classification_report_stacking)



# Train Word2Vec model on the entire dataset
word2vec_model = Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=1, workers=4)

# Function to get the vector representation of a sentence
def get_vector(sentence, model):
    vector = [model.wv[word] for word in sentence if word in model.wv]
    return sum(vector) / len(vector) if vector else [0] * model.vector_size

# Apply Word2Vec to the entire dataset
df['text_vector'] = tokenized_text.apply(lambda x: get_vector(x, word2vec_model))

# Split the dataset into training and testing sets for Word2Vec
X_train_w2v, X_test_w2v, y_train_w2v, y_test_w2v = train_test_split(df['text_vector'], df['label'], test_size=0.2, random_state=42)

# Train a classifier (e.g., Logistic Regression) on the Word2Vec transformed data
from sklearn.linear_model import LogisticRegression

logistic_regression_w2v = LogisticRegression()
logistic_regression_w2v.fit(X_train_w2v.tolist(), y_train_w2v)

# Predictions on the Word2Vec test set
predictions_w2v = logistic_regression_w2v.predict(X_test_w2v.tolist())

# Evaluate the performance of Word2Vec
accuracy_w2v = accuracy_score(y_test_w2v, predictions_w2v)
classification_report_w2v = classification_report(y_test_w2v, predictions_w2v)

print("\nWord2Vec Classifier:")
print("Accuracy:", accuracy_w2v)
print("Classification Report:\n", classification_report_w2v)
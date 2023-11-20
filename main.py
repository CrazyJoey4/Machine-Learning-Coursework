import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


def load_data(file_path):
    df = pd.read_csv(file_path, encoding='latin-1')
    df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
    df = df.rename(columns={"v1": "label", "v2": "text"})
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df


# Function to plot word frequency
def plot_word_frequency_pie(text, title):
    words = text.split()
    word_freq = nltk.FreqDist(words)
    plt.figure(figsize=(8, 8))
    labels, sizes = zip(*word_freq.most_common(5))  # Display top 5 words
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.show()

# Function to visualize dataset status
def visualize_dataset_status_pie(df):
    # Pie chart for label distribution
    plt.figure(figsize=(8, 8))
    labels = 'Ham', 'Spam'
    sizes = df['label'].value_counts().values
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#66b3ff', '#99ff99'])
    plt.title('Distribution of Labels in the Dataset')
    plt.show()

    # Pie chart for word frequency in the entire dataset
    all_text = ' '.join(df['text'])
    plot_word_frequency_pie(all_text, 'Word Frequency in the Entire Dataset')

    # Pie chart for word frequency in 'ham' messages
    ham_text = ' '.join(df[df['label'] == 0]['text'])
    plot_word_frequency_pie(ham_text, 'Word Frequency in "Ham" Messages')

    # Pie chart for word frequency in 'spam' messages
    spam_text = ' '.join(df[df['label'] == 1]['text'])
    plot_word_frequency_pie(spam_text, 'Word Frequency in "Spam" Messages')


def preprocess_text(text):
    text = re.sub("[^a-zA-Z]", " ", text.lower())
    words = text.split()
    words = [word for word in words if word not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


def preprocess_data(df):
    df['text'] = df['text'].apply(preprocess_text)
    return df


def split_data(df, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def train_naive_bayes(X_train_tfidf, y_train, alpha=1.0):
    nb_classifier = MultinomialNB(alpha=alpha)
    nb_classifier.fit(X_train_tfidf, y_train)
    return nb_classifier


def train_svm(X_train_tfidf, y_train, kernel='linear', C=1.0):
    svm_classifier = SVC(kernel=kernel, C=C)
    svm_classifier.fit(X_train_tfidf, y_train)
    return svm_classifier


def evaluate_classifier(classifier, X_test_tfidf, y_test):
    predictions = classifier.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)
    classification_report_str = classification_report(y_test, predictions)
    return accuracy, classification_report_str


def optimize_naive_bayes(X_train_tfidf, y_train):
    nb_classifier = MultinomialNB()
    nb_params = {'alpha': [0.1, 0.5, 1.0, 2.0]}
    nb_grid_search = GridSearchCV(nb_classifier, nb_params, cv=5, scoring='accuracy')
    nb_grid_search.fit(X_train_tfidf, y_train)
    best_alpha = nb_grid_search.best_params_['alpha']

    # Fit the optimized model with the best hyperparameters
    optimized_nb_classifier = MultinomialNB(alpha=best_alpha)
    optimized_nb_classifier.fit(X_train_tfidf, y_train)

    return optimized_nb_classifier, best_alpha


def optimize_svm(X_train_tfidf, y_train):
    svm_classifier = SVC(kernel='linear')
    svm_params = {'C': [0.1, 1, 10]}
    svm_grid_search = GridSearchCV(svm_classifier, svm_params, cv=5, scoring='accuracy')
    svm_grid_search.fit(X_train_tfidf, y_train)
    best_C = svm_grid_search.best_params_['C']

    # Fit the optimized model with the best hyperparameters
    optimized_svm_classifier = SVC(kernel='linear', C=best_C)
    optimized_svm_classifier.fit(X_train_tfidf, y_train)

    return optimized_svm_classifier, best_C


# Load data
df = load_data('spam.csv')

# Preprocess data and Visualized
df = preprocess_data(df)
visualize_dataset_status_pie(df)

# Split the dataset
X_train, X_test, y_train, y_test = split_data(df)

# Convert text data to numerical features using CountVectorizer
count_vectorizer = CountVectorizer()
X_train_counts = count_vectorizer.fit_transform(X_train)
X_test_counts = count_vectorizer.transform(X_test)

# Convert term frequency to TF-IDF
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)



# Train Naive Bayes Classifier
naive_bayes_classifier = train_naive_bayes(X_train_tfidf, y_train)

# Evaluate Naive Bayes Classifier
accuracy_nb, classification_report_nb = evaluate_classifier(naive_bayes_classifier, X_test_tfidf, y_test)

print("Naive Bayes Classifier:")
print("Accuracy:", accuracy_nb)
print("Classification Report:\n", classification_report_nb)



# Optimize Naive Bayes Classifier
optimized_nb_classifier, best_alpha_nb = optimize_naive_bayes(X_train_tfidf, y_train)

# Evaluate Optimized Naive Bayes Classifier
accuracy_optimized_nb, classification_report_optimized_nb = evaluate_classifier(optimized_nb_classifier, X_test_tfidf, y_test)

print("\nOptimized Naive Bayes Classifier with alpha =", best_alpha_nb)
print("Accuracy:", accuracy_optimized_nb)
print("Classification Report:\n", classification_report_optimized_nb)



# Train SVM Classifier
svm_classifier = train_svm(X_train_tfidf, y_train)

# Evaluate SVM Classifier
accuracy_svm, classification_report_svm = evaluate_classifier(svm_classifier, X_test_tfidf, y_test)

print("\nSupport Vector Machines (SVM) Classifier:")
print("Accuracy:", accuracy_svm)
print("Classification Report:\n", classification_report_svm)



# Optimize SVM Classifier
optimized_svm_classifier, best_C_svm = optimize_svm(X_train_tfidf, y_train)

# Evaluate Optimized SVM Classifier
accuracy_optimized_svm, classification_report_optimized_svm = evaluate_classifier(optimized_svm_classifier, X_test_tfidf, y_test)

print("\nOptimized SVM Classifier with C =", best_C_svm)
print("Accuracy:", accuracy_optimized_svm)
print("Classification Report:\n", classification_report_optimized_svm)



# For testing
def preprocess_input(text):
    # Your preprocessing logic here (similar to what you did with the training data)
    text = re.sub("[^a-zA-Z]", " ", text.lower())
    words = text.split()
    words = [word for word in words if word not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


def classify_input(input_text, classifier):
    # Preprocess the entire input text
    preprocessed_input = preprocess_input(input_text)

    # Convert the preprocessed text to numerical features
    input_counts = count_vectorizer.transform([preprocessed_input])
    input_tfidf = tfidf_transformer.transform(input_counts)

    # Make predictions using the classifier
    prediction = classifier.predict(input_tfidf)

    return prediction[0]

def classify_input_optimized(input_text, classifier, count_vectorizer, tfidf_transformer):
    # Preprocess the entire input text
    preprocessed_input = preprocess_input(input_text)

    # Convert the preprocessed text to numerical features
    input_counts = count_vectorizer.transform([preprocessed_input])
    input_tfidf = tfidf_transformer.transform(input_counts)

    # Make predictions using the optimized classifier
    prediction = classifier.predict(input_tfidf)

    return prediction[0]


# Example usage with a loop
while True:
    user_input = input("| Naive Bayes Classifier |\nEnter a sentence to classify (enter 'x' to exit): ")

    if user_input.lower() == 'x':
        break

    result = classify_input(user_input, naive_bayes_classifier)

    if result == 0:
        print("Predicted: Ham")
    else:
        print("Predicted: Spam")


print("\n-----------------------------------------------------------------\n")

# 2nd Example usage with a loop
while True:
    user_input = input("\n| Naive Bayes Classifier Optimized |\nEnter a sentence to classify (enter 'x' to exit): ")

    if user_input.lower() == 'x':
        break

    result_optimized = classify_input_optimized(user_input, optimized_nb_classifier, count_vectorizer, tfidf_transformer)

    if result_optimized == 0:
        print("Predicted: Ham")
    else:
        print("Predicted: Spam")

print("\n-----------------------------------------------------------------\n")

# 3rd Example usage with a loop
while True:
    user_input = input("\n\n| SVM Classifier |\nEnter a sentence to classify (enter 'x' to exit): ")

    if user_input.lower() == 'x':
        break

    result = classify_input(user_input, svm_classifier)

    if result == 0:
        print("Predicted: Ham")
    else:
        print("Predicted: Spam")

print("-----------------------------------------------------------------\n")

# 4th Example usage with a loop
while True:
    user_input = input("\n\n| SVM Classifier Optimized |\nEnter a sentence to classify (enter 'x' to exit): ")

    if user_input.lower() == 'x':
        break

    result_optimized = classify_input_optimized(user_input, optimized_svm_classifier, count_vectorizer, tfidf_transformer)

    if result_optimized == 0:
        print("Predicted: Ham")
    else:
        print("Predicted: Spam")
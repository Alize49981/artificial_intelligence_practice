import pandas as pd
data = {
    "text": [
        "Win money now",
        "Free iPhone offer",
        "Call me later",
        "Let's meet tomorrow",
        "Claim your prize",
        "Important meeting today"
    ],
    "label": [1, 1, 0, 0, 1, 0]  
}

df = pd.DataFrame(data)

print(df)

print(df.head())
print(df.info())
#Split Data (Train/Test)
from sklearn.model_selection import train_test_split

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Text Preprocessing + Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer
#Convert Text → Numbers
vectorizer = CountVectorizer()

X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)
#View the Vocabulary
print(vectorizer.get_feature_names_out())
#See Numerical Data
print(X_train_vectors.toarray())

#Train the Spam Detection Model (Naive Bayes)
from sklearn.naive_bayes import MultinomialNB
#Train the Model
model = MultinomialNB()

# Train using vectorized data
model.fit(X_train_vectors, y_train)
#Make Predictions
y_pred = model.predict(X_test_vectors)

print(y_pred)
#Test with New Messages
sample_messages = [
    "Congratulations! You won a prize",
    "Let's have a meeting tomorrow"
]

sample_vectors = vectorizer.transform(sample_messages)
predictions = model.predict(sample_vectors)

for msg, pred in zip(sample_messages, predictions):
    print(msg, "-> Spam" if pred == 1 else "Not Spam")

#Evaluate Your AI Model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
#Classification Report
print(classification_report(y_test, y_pred))

#Use TF-IDF (Better than CountVectorizer)
from sklearn.feature_extraction.text import TfidfVectorizer
#Replace CountVectorizer
vectorizer = TfidfVectorizer()

X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)
#Improve the Model
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train_vectors, y_train)
#test examples
messages = [
    "Congratulations! You have won $1000",
    "Reminder: meeting at 10am",
    "Free entry in a contest now",
    "Are we still on for lunch?"
]

vectors = vectorizer.transform(messages)
predictions = model.predict(vectors)

for msg, pred in zip(messages, predictions):
    print(msg, "-> Spam" if pred else "Not Spam")

#Remove Stopwords
vectorizer = TfidfVectorizer(stop_words='english')
#Use N-grams
vectorizer = TfidfVectorizer(ngram_range=(1,2))
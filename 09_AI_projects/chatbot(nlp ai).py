#create chatbot logic
def chatbot(user_input):
    user_input = user_input.lower()

    if "hello" in user_input:
        return "Hello! How can I help you today?"

    elif "name" in user_input:
        return "I am your AI chatbot 🤖"

    elif "study" in user_input:
        return "Keep studying consistently, you are improving!"

    elif "python" in user_input:
        return "Python is great for AI and machine learning!"

    else:
        return "Sorry, I don't understand that yet."
#test chatbot
print(chatbot("Hello"))
print(chatbot("What is your name?"))
print(chatbot("I want to study AI"))
print(chatbot("Tell me about Python"))
#Make It Interactive (Loop)
print("AI Chatbot is running... Type 'exit' to stop.")

while True:
    user = input("You: ")

    if user.lower() == "exit":
        print("Bot: Goodbye 👋")
        break

    response = chatbot(user)
    print("Bot:", response)
#Smart ML Chatbot (TF-IDF + Classification)
#create training data(intent)
import pandas as pd

data = {
    "text": [
        "hello",
        "hi there",
        "good morning",
        "what is your name",
        "who are you",
        "tell me about python",
        "what is AI",
        "bye",
        "see you later",
        "thanks"
    ],
    "intent": [
        "greeting",
        "greeting",
        "greeting",
        "identity",
        "identity",
        "python_info",
        "ai_info",
        "goodbye",
        "goodbye",
        "thanks"
    ]
}

df = pd.DataFrame(data)
print(df)
#Convert Text → Numbers (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])

y = df["intent"]

#train ai model
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X, y)
#create response system
responses = {
    "greeting": "Hello 👋! How can I help you?",
    "identity": "I am your AI chatbot 🤖",
    "python_info": "Python is great for AI and machine learning.",
    "ai_info": "AI helps machines think and learn like humans.",
    "goodbye": "Goodbye 👋! Have a nice day.",
    "thanks": "You're welcome 😊"
}
#predict user intent
def chatbot(user_input):
    input_vector = vectorizer.transform([user_input])
    intent = model.predict(input_vector)[0]
    
    return responses.get(intent, "Sorry, I don't understand.")
#test my ai chatbot
print(chatbot("hello there"))
print(chatbot("tell me about AI"))
print(chatbot("what is python"))
print(chatbot("bye"))
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
import spacy

# Initialize Flask App
app = Flask(__name__)

# Load SpaCy model for NLP
nlp = spacy.load("en_core_web_sm")

# Load Dataset
data = pd.read_csv('questions_dataset.csv')  # Columns: 'question', 'category'

# Preprocessing Function
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Preprocess Questions
data['processed_question'] = data['question'].apply(preprocess_text)

# Encode Categories
label_encoder = LabelEncoder()
data['category_encoded'] = label_encoder.fit_transform(data['category'])

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    data['processed_question'], data['category_encoded'], test_size=0.2, random_state=42
)

# Build Model Pipeline
model = make_pipeline(TfidfVectorizer(), LinearSVC())
model.fit(X_train, y_train)

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get('question', '')
    if not user_input:
        return jsonify({'error': 'No question provided.'}), 400

    # Preprocess and Predict
    processed_input = preprocess_text(user_input)
    category_encoded = model.predict([processed_input])[0]
    category = label_encoder.inverse_transform([category_encoded])[0]

    # Predefined Responses
    responses = {
        "1": "Response for category 1.",
        "2": "Response for category 2.",
        # ... Add all 100 responses
    }

    response = responses.get(category, "Sorry, I don't understand your question.")
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)

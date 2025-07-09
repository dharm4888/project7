from flask import Flask, request, render_template, redirect, url_for
import joblib
import re
import numpy as np

app = Flask(__name__)

# Load pre-trained model and vectorizer
model = joblib.load(r"https://github.com/dharm4888/project7/blob/main/disaster_classifier.pkl")
tfidf = joblib.load(r'https://github.com/dharm4888/project7/blob/main/tfidf_vectorizer.pkl')

# Text cleaning function (must match preprocessing during training)
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|@\w+|[^\w\s]', '', text)  # Remove URLs/mentions/punctuation
    return text.lower().strip()

@app.route('/')
def home():
    """Render the main input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission and display prediction results"""
    # Get tweet text from form
    tweet_text = request.form['tweet']
    
    # Clean and vectorize the text
    cleaned_text = clean_text(tweet_text)
    vector = tfidf.transform([cleaned_text])
    
    # Make prediction and get confidence score
    prediction = model.predict(vector)[0]
    confidence = np.max(model.predict_proba(vector)) * 100  # Get highest class probability
    result = "Disaster" if prediction == 1 else "Non-Disaster"
    
    # Render results page with context
    return render_template(
        'result.html',
        prediction=result,
        tweet_text=tweet_text,
        confidence=f"{confidence:.1f}"
    )

@app.errorhandler(404)
def page_not_found(e):
    """Custom 404 error handling"""
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True)  # Set debug=False in production

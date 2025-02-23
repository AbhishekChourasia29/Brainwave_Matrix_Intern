from flask import Flask, render_template, request, jsonify
import pickle
import requests
import time

app = Flask(__name__)

# Load your pre-trained Fake News model and TF-IDF vectorizer
with open('model_updated.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('vectorizer_updated.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Set your NewsData.io API key (provided)
NEWS_API_KEY = 'pub_713418467b9d3ab7b1836ee003a5193b34e04'

# Simple in-memory cache for headlines (refresh every hour)
NEWS_CACHE = {"timestamp": 0, "headlines": []}

def fetch_news_headlines():
    current_time = time.time()
    # Use cached headlines if fetched within the last hour
    if current_time - NEWS_CACHE["timestamp"] < 3600 and NEWS_CACHE["headlines"]:
        return NEWS_CACHE["headlines"]
    
    # Using NewsData.io API endpoint for Indian headlines
    url = f"https://newsdata.io/api/1/news?apikey={NEWS_API_KEY}&country=in&language=en&category=top"
    response = requests.get(url)
    data = response.json()
    headlines = []
    if data.get("status") == "success":
        for article in data.get("results", []):
            headlines.append({
                "title": article.get("title"),
                "url": article.get("link")
            })
    else:
        print("Error fetching headlines:", data.get("message"))
    # Cache the top 20 headlines (if more are returned)
    NEWS_CACHE["timestamp"] = current_time
    NEWS_CACHE["headlines"] = headlines[:20]
    return NEWS_CACHE["headlines"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    news_text = request.form.get('news')
    if news_text:
        news_vector = vectorizer.transform([news_text])
        prediction = model.predict(news_vector)[0]
        probabilities = model.predict_proba(news_vector)[0]
        # Assuming model.classes_ returns ["FAKE", "REAL"]
        fake_index = list(model.classes_).index("FAKE")
        real_index = list(model.classes_).index("REAL")
        fake_prob = probabilities[fake_index]
        real_prob = probabilities[real_index]
        return jsonify({
            "prediction": prediction, 
            "real_confidence": round(real_prob * 100, 2),
            "fake_confidence": round(fake_prob * 100, 2)
        })
    else:
        return jsonify({"error": "No news text provided."}), 400

@app.route('/api/news', methods=['GET'])
def get_news():
    headlines = fetch_news_headlines()
    return jsonify(headlines)

if __name__ == '__main__':
    app.run(debug=True)

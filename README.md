# Barinwave Matrix Internship AI/ML Intern Task -1 by Abhishek Chourasia

# Fake News Detector Project Report

This project is an end-to-end solution using machine learning and deep learning techniques to classify news articles as either **REAL** or **FAKE**. It includes real-time predictions, dynamic news headlines, and a responsive user interface, and is deployed locally using a Flask application.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset and Model Development](#dataset-and-model-development)
  - [Data Collection and Pre-processing](#data-collection-and-pre-processing)
  - [Model Training and Evaluation](#model-training-and-evaluation)
- [Data Visualization](#data-visualization)
- [Application Architecture and Implementation](#application-architecture-and-implementation)
  - [Backend – Flask Application](#backend--flask-application)
  - [Frontend – HTML, CSS, and JavaScript](#frontend--html-css-and-javascript)
- [Deployment on Local Host](#deployment-on-local-host)
- [System Architecture Diagram](#system-architecture-diagram)
- [Conclusion and Future Enhancements](#conclusion-and-future-enhancements)
- [Reference: Model Training Code](#reference-model-training-code)

## Project Overview

The Fake News Detector project is designed to classify news articles as **REAL** or **FAKE** using advanced machine learning and deep learning methods. The system offers:

- **Real-Time Predictions:** Users input news text to receive predictions along with confidence percentages.
- **Dynamic News Headlines:** Retrieves and displays top news headlines from India via an external API.
- **Responsive User Interface:** Built with HTML, CSS, and JavaScript for an optimal experience on mobile and desktop devices.
- **Local Deployment:** Runs on a local host using Flask, making it ideal for development, testing, and future enhancements.

## Dataset and Model Development

### Data Collection and Pre-processing

- **Dataset Sources:**
  - A dataset containing fake news articles.
  - A dataset containing true news articles.
- **Pre-processing Steps:**
  - **Labelling:** Assign labels ("FAKE" for fake news and "REAL" for true news).
  - **Combining Data:** Merge the datasets into a single data frame.
  - **Feature Extraction:** Use the text column (optionally combining title and text) and convert it into numerical features using a TF-IDF vectorizer (excluding English stop words).

### Model Training and Evaluation

- **Data Splitting:** 
  - Split the dataset into training and testing sets with an 80/20 ratio.
- **Vectorization:** 
  - Transform text data using the TF-IDF vectorizer.
- **Classifier:** 
  - Train a Logistic Regression classifier (with a maximum of 1000 iterations) on the training data.
- **Evaluation:** 
  - Evaluate the model’s accuracy on the test set and print the classification report.
- **Model Persistence:** 
  - Save the trained model as `model_updated.pkl` and the TF-IDF vectorizer as `vectorizer_updated.pkl` using Python's pickle module.

## Data Visualization

The project includes several visualizations to assess model performance:

- **Confusion Matrix:** A heatmap displaying the number of correct and incorrect predictions for each class.
- **ROC Curve:** Plots the True Positive Rate (TPR) versus the False Positive Rate (FPR) along with the Area Under the Curve (AUC) value.
- **Precision-Recall Curve:** Illustrates the trade-off between precision and recall, which is especially useful for imbalanced datasets.
- **Distribution of True Labels:** A bar chart showing the count of each label in the test set.
- **Distribution of Predicted Labels:** A bar chart comparing predicted labels with true labels to assess potential biases.

## Application Architecture and Implementation

### Backend – Flask Application

- **Routes:**
  - `/` : Renders the main HTML page.
  - `/api/predict` : Accepts POST requests with news text, transforms the text using the TF-IDF vectorizer, and returns predictions with confidence percentages.
  - `/api/news` : Fetches the top headlines from the NewsData.io API (for Indian news) and caches results for one hour.
- **Model Loading:** 
  - Loads `model_updated.pkl` and `vectorizer_updated.pkl` at startup to provide immediate predictions.

### Frontend – HTML, CSS, and JavaScript

- **User Interface:**
  - **Hero Section:** Features a dynamic video background or image for an engaging introduction.
  - **Fake News Detector Section:** Contains a live clock, a text area for news input, a submit button, animated progress bars for prediction display, and a history log.
  - **Headlines Section:** Dynamically displays top news headlines fetched from the backend.
  - **Why Detector Section:** Explains the importance of fake news detection.
- **Interactivity:** 
  - Managed via JavaScript, including debouncing user input to avoid overwhelming the server, updating the UI with predictions, and periodically refreshing news headlines.

## Deployment on Local Host

### Steps to Deploy

1. **Environment Setup:**
   - Install Python and the required libraries (Flask, requests, scikit-learn, etc.).
   - Ensure all project files (e.g., `app.py`, `index.html`, model pickle files, and static assets) are placed in the appropriate directories.
2. **Running the Application:**
   - Open a terminal in the project folder.
   - Run the Flask server with:
     ```bash
     python app.py
     ```
   - The application will run at [http://localhost:5000](http://localhost:5000).
3. **Accessing the Application:**
   - Open your web browser and navigate to [http://localhost:5000](http://localhost:5000) to use the Fake News Detector.

## System Architecture Diagram

Include your system architecture diagram image in the repository. If generated using a tool (e.g., matplotlib), save the image as `architecture_diagram.png` and reference it here:

![System Architecture Diagram](architecture_diagram.png)

## Conclusion and Future Enhancements

### Conclusion

This project demonstrates a comprehensive solution for fake news detection by:

- Processing and vectorizing news text.
- Utilizing a Logistic Regression model to classify news as **REAL** or **FAKE**.
- Providing real-time predictions and dynamic news headline updates.
- Deploying a user-friendly interface via Flask for local development and testing.

### Future Enhancements

- **Model Optimization:** Explore more advanced machine learning or deep learning models.
- **Scalability:** Containerize the application using Docker and deploy on cloud platforms.
- **Enhanced UI/UX:** Improve the frontend experience with modern frameworks like React or Vue.js.
- **Data Persistence:** Implement a database for logging predictions and user interactions.
- **User Feedback:** Integrate user feedback mechanisms to continuously refine the model.

## Reference: Model Training Code

Below is an excerpt of the code used for data loading, pre-processing, model training, evaluation, and visualization:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
fake = pd.read_csv('fake.csv')
real = pd.read_csv('true.csv')

# Label datasets
fake['label'] = 'FAKE'
real['label'] = 'REAL'

# Combine datasets
data = pd.concat([fake, real], ignore_index=True)

# Use text column as features and label as target
X = data['text']
y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Evaluation
predictions = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)
report = classification_report(y_test, predictions, target_names=["FAKE", "REAL"])
print(report)

# Save model and vectorizer
with open('model_updated.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('vectorizer_updated.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

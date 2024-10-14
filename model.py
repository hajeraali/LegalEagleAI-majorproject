import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load the dataset
lawyer_data = pd.read_csv('lawyers_dataset.csv')

# Combine relevant columns into a single 'info' column to be used as training data
lawyer_data['info'] = lawyer_data['Practice_area'] + " " + lawyer_data['Designation'] + " " + lawyer_data['Client_reviews']

# Create TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

# Fit the vectorizer on the 'info' column
tfidf_vectorizer.fit(lawyer_data['info'])

# Save the trained model for future use
joblib.dump(tfidf_vectorizer, 'lawyer_recommendation_model.pkl')

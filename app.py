import os
import re
import nltk
import pandas as pd
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from PyPDF2 import PdfReader
from docx import Document

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load dataset
data = pd.read_csv('lawyers_dataset.csv')
print(data.head())  # Verify the loaded data
print(data.columns)  # Check the column names

# Load a pre-trained model for text classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define practice areas
practice_areas = [
    'Corporate Lawyer',
    'Civil Lawyer',
    'Criminal Lawyer',
    'Constitutional Lawyer',
    'Administrative Lawyer',
    'Business Lawyer',
    'Intellectual Property Lawyer',
    'Patent Lawyer',
    'Trademark Lawyer',
    'Copyright Lawyer',
    'Environmental Lawyer',
    'Banking and Finance Lawyer',
    'Bankruptcy Lawyer',
    'Civil Rights Lawyer',
    'Family Lawyer',
    'Employment Lawyer',
    'Immigration Lawyer',
    'Personal Injury Lawyer',
    'Tax Lawyer',
    'Military Lawyer',
    'International Lawyer',
    'Municipal Lawyer',
    'Animal Lawyer',
    'Education Lawyer',
    'Elder Lawyer',
    'Entertainment Lawyer',
    'Sports Lawyer',
    'Securities Lawyer',
    'Health Lawyer',
    'Real Estate Lawyer',
    'Maritime Lawyer',
    'Labor Lawyer'
]


# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_query(query):
    """Preprocess the input query by cleaning, tokenizing, and stemming."""
    query = re.sub(r'[^\w\s]', '', query.lower())
    tokens = nltk.word_tokenize(query)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def extract_text_from_file(file):
    """Extract text from different file types."""
    if file.filename.endswith('.txt'):
        return file.read().decode('utf-8')
    elif file.filename.endswith('.pdf'):
        reader = PdfReader(file)
        text = ''
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
        return text
    elif file.filename.endswith('.docx'):
        doc = Document(file)
        text = ''
        for paragraph in doc.paragraphs:
            text += paragraph.text + '\n'
        return text
    return None

def allowed_file(filename):
    """Check if the file type is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'txt', 'pdf', 'docx'}

def recommend_lawyers(query):
    # Preprocess the query
    cleaned_query = preprocess_query(query)
    print(f"Cleaned Query: {cleaned_query}")  # Debug print

    if not cleaned_query.strip():
        raise ValueError("The query is empty after preprocessing.")

    # Use zero-shot classification to identify the practice area
    classification = classifier(cleaned_query, practice_areas)
    print(f"Classification: {classification}")  # Debug print
    
    # Get the best practice area
    recommended_practice_area = classification['labels'][0]
    print(f"Recommended Practice Area: {recommended_practice_area}")  # Debug print

    # Filter lawyers based on the recommended practice area
    recommendations = data[data['Practice_area'] == recommended_practice_area]

    # Check if recommendations are empty and provide feedback
    if recommendations.empty:
        print("No lawyers found in the recommended area. Searching by keywords...")
        keywords = cleaned_query.split()
        for index, row in data.iterrows():
            combined_fields = ' '.join(row.astype(str))
            if any(keyword in combined_fields for keyword in keywords):
                recommendations = recommendations.append(row)

    print(f"Recommendations found: {recommendations}")  # Debug print
    return recommendations[['Sr_No.', 'Lawyer_name', 'Practice_area', 'Firm_name', 
                             'Firm_size', 'Target_audience', 'Designation', 
                             'Years_of_Experience', 'Total_cases', 
                             'Successful_cases', 'Affiliation', 
                             'Client_reviews', 'Nominal_fees_per_hearing', 
                             'Bar_Council_ID']]

@app.route('/', methods=['GET', 'POST'])
def index():
    recommended_lawyers = None
    if request.method == 'POST':
        user_query = None
        if 'query' in request.form:  # Typed query
            user_query = request.form['query']
        elif 'file' in request.files:  # Document upload
            file = request.files['file']
            if file and allowed_file(file.filename):
                user_query = extract_text_from_file(file)

        if user_query:
            recommended_lawyers = recommend_lawyers(user_query)

    return render_template('index.html', recommended_lawyers=recommended_lawyers)

if __name__ == "__main__":
    app.run(debug=True)

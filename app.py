import os
import re
import nltk
import pandas as pd
from flask import Flask, render_template, request
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import docx

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load dataset
data = pd.read_csv('lawyers_dataset.csv')

# Load a pre-trained model for text classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define practice areas
practice_areas = [
    # Existing practice areas
    
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
    """Extract text from a PDF or DOCX file."""
    if file.filename.endswith('.pdf'):
        reader = PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text
    elif file.filename.endswith('.docx'):
        doc = docx.Document(file)
        text = '\n'.join([para.text for para in doc.paragraphs])
        return text
    return ''

@app.route('/', methods=['GET', 'POST'])
def index():
    lawyer_recommendations = None
    sort_order = None  # Initialize sort order variable
    if request.method == 'POST':
        user_query = None
        min_price = None
        max_price = None

        # Handle typed query
        if 'query' in request.form and request.form['query']:
            user_query = request.form['query']
        
        # Handle document upload
        if 'upload' in request.files and request.files['upload']:
            file = request.files['upload']
            if file and (file.filename.endswith('.pdf') or file.filename.endswith('.docx')):
                document_text = extract_text_from_file(file)
                user_query = document_text  # Set user_query to the text from the document
        
        # Get the price range if provided
        if 'min_price' in request.form and 'max_price' in request.form:
            min_price = request.form.get('min_price')
            max_price = request.form.get('max_price')

        # Get the sort order if provided
        sort_order = request.form.get('sort_order')

        if user_query:
            lawyer_recommendations = recommend_lawyers(user_query, min_price, max_price, sort_order)

    return render_template('index.html', recommended_lawyers=lawyer_recommendations)

def recommend_lawyers(query, min_price=None, max_price=None, sort_order=None):
    # Preprocess the query
    cleaned_query = preprocess_query(query)

    if not cleaned_query.strip():
        raise ValueError("The query is empty after preprocessing.")

    # Use zero-shot classification to identify the practice areas
    classification = classifier(cleaned_query, practice_areas)
    recommended_practice_area = classification['labels'][0]

    # Filter lawyers based on the recommended practice area
    recommendations = data[data['Practice_area'] == recommended_practice_area]

    # Ensure 'Nominal_fees_per_hearing' is of numeric type
    recommendations.loc[:, 'Nominal_fees_per_hearing'] = pd.to_numeric(recommendations['Nominal_fees_per_hearing'], errors='coerce')

    # Filter by nominal fees per hearing if a range is provided
    if min_price is not None and max_price is not None:
        try:
            min_price = float(min_price)
            max_price = float(max_price)

            recommendations = recommendations[
                (recommendations['Nominal_fees_per_hearing'] >= min_price) &
                (recommendations['Nominal_fees_per_hearing'] <= max_price)
            ]
        except ValueError:
            # Handle conversion errors
            pass

    # Sort recommendations based on user selection
    if sort_order == 'low_to_high':
        recommendations = recommendations.sort_values(by='Nominal_fees_per_hearing', ascending=True)
    elif sort_order == 'high_to_low':
        recommendations = recommendations.sort_values(by='Nominal_fees_per_hearing', ascending=False)

    return recommendations

if __name__ == '__main__':
    app.run(debug=True)

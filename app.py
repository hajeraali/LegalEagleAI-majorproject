import os
import re
import nltk
import pandas as pd
from flask import Flask, render_template, request, jsonify, g
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import docx
from datetime import datetime, timedelta
import psycopg2
from psycopg2 import sql
from flask_mailman import Mail
from flask_mailman.message import EmailMessage
import google.generativeai as genai

from check_env import init_mail, send_email 
from dotenv import load_dotenv

load_dotenv()  
app = Flask(__name__)
init_mail(app)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

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
    try:
        if file.filename.endswith('.pdf'):
            reader = PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() or ''  # Handle empty page text
            return text.strip()
        elif file.filename.endswith('.docx'):
            doc = docx.Document(file)
            text = '\n'.join([para.text for para in doc.paragraphs])
            return text.strip()
    except Exception as e:
        print(f"Error extracting text from file: {e}")  # Debugging log
    return ''


@app.route('/recommend_lawyers', methods=['GET', 'POST'])
def recommend_lawyers_route():
    lawyer_recommendations = None
    error_message = None
    sort_order = None  # Initialize sort order variable

    if request.method == 'POST':
        user_query = None
        min_price = None
        max_price = None
        location = None  # Initialize location variable

        try:
            # Handle typed query
            if 'query' in request.form and request.form['query']:
                user_query = request.form['query']
            
            # Handle document upload
            if 'upload' in request.files and request.files['upload']:
                file = request.files['upload']
                if file and (file.filename.endswith('.pdf') or file.filename.endswith('.docx')):
                    document_text = extract_text_from_file(file)
                    if document_text:
                        user_query = document_text
                    else:
                        error_message = "Failed to extract text from the uploaded document."
            
            # Get the price range if provided
            if 'min_price' in request.form and 'max_price' in request.form:
                min_price = request.form.get('min_price')
                max_price = request.form.get('max_price')

            # Get the sort order if provided
            sort_order = request.form.get('sort_order')

            # Get the location if provided
            if 'location' in request.form and request.form['location']:
                location = request.form['location']

            if user_query:
                lawyer_recommendations = recommend_lawyers(user_query, min_price, max_price, sort_order, location)
            else:
                error_message = "No query provided. Please type a query or upload a valid document."
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            print(error_message)  # Debug log

    return render_template(
        'recommend_lawyers.html',
        recommended_lawyers=lawyer_recommendations,
        error_message=error_message,
    )

    
# Configure the API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")
genai.configure(api_key=GOOGLE_API_KEY)
@app.route("/api/generate", methods=["POST"])
def generate():
    try:
        # Get the question from the incoming JSON
        data = request.json
        user_question = data.get("question", "").strip().lower()

        # If the question is empty
        if not user_question:
            return jsonify({"error": "Please provide a valid question."}), 400

        # Check if the user is asking for lawyer recommendations
        if "recommend lawyer" in user_question or "find lawyer" in user_question:
            return jsonify({
                "response": 'I can help you find a lawyer! <a href="/client_login">Recommend</a>'
            })

        # Set up the chatbot prompt
        summarization_prompt = f"You are a chatbot that gives legal advice only if asked. Otherwise, answer the following question in simple answers and in no more than 70 words: {user_question}"

        # Generate a response using the AI model
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",  # Use a valid model name
        )

        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(summarization_prompt)

        # Return the AI-generated response
        return jsonify({"response": response.text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
load_dotenv()    
# Load Firebase config and make it globally available before each request
@app.before_request
def load_firebase_config():
    g.firebase_config = {
        'firebase_api_key': os.getenv('FIREBASE_API_KEY'),
        'firebase_auth_domain': os.getenv('FIREBASE_AUTH_DOMAIN'),
        'firebase_project_id': os.getenv('FIREBASE_PROJECT_ID'),
        'firebase_storage_bucket': os.getenv('FIREBASE_STORAGE_BUCKET'),
        'firebase_messaging_sender_id': os.getenv('FIREBASE_MESSAGING_SENDER_ID'),
        'firebase_app_id': os.getenv('FIREBASE_APP_ID')
    }
    
@app.route('/')
def index():
    return render_template('index.html')  # The new landing page

@app.route('/client_signup')
def client_signup():
    return render_template('client_signup.html', **g.firebase_config)

@app.route('/client_login')
def client_login():
    return render_template('client_login.html', **g.firebase_config)

@app.route('/afterlogin')
def afterlogin():
    return render_template('afterlogin.html', **g.firebase_config)

@app.route('/lawyer_login.html')
def lawyer_login():
    return render_template('lawyer_login.html', **g.firebase_config)

@app.route('/lawyer_signup.html')
def lawyer_signup():
    return render_template('lawyer_signup.html', **g.firebase_config)

@app.route('/dashboard.html')
def dashboard():
    return render_template('dashboard.html', **g.firebase_config)


def recommend_lawyers(query, min_price=None, max_price=None, sort_order=None, location=None):
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

    # Filter recommendations by location if provided
    if location:
        recommendations = recommendations[recommendations['Location'].str.contains(location, case=False, na=False)]

    # Sort recommendations based on user selection
    if sort_order == 'low_to_high':
        recommendations = recommendations.sort_values(by='Nominal_fees_per_hearing', ascending=True)
    elif sort_order == 'high_to_low':
        recommendations = recommendations.sort_values(by='Nominal_fees_per_hearing', ascending=False)

    return recommendations


# Database configuration (replace karo with ur actual database credentials)
DB_HOST = 'localhost'
DB_NAME = 'mydatabase'
DB_USER = 'postgres'
DB_PASSWORD = '123'

# Connect to PostgreSQL
def connect_db():
    return psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

# Function to create the appointments table if it doesn’t exist
def create_table():
    conn = connect_db()
    cur = conn.cursor()
    create_table_query = """
    CREATE TABLE IF NOT EXISTS public.clientappointments (
        id SERIAL PRIMARY KEY,
        appointment_date DATE NOT NULL,
        client_name VARCHAR(100) NOT NULL,
        client_email VARCHAR(100) NOT NULL,
        appointment_time TIME NOT NULL,
        case_details TEXT NOT NULL,
        lawyer_name VARCHAR(100) NOT NULL
    );
    """

    cur.execute(create_table_query)
    conn.commit()
    cur.close()
    conn.close()

# Call create_table when the app starts
create_table()
@app.route('/booking.html')
def booking():
    lawyer_name = request.args.get('lawyer')  # Get lawyer name from URL parameters
    return render_template('booking.html', lawyer_name=lawyer_name)

# Route to handle form submissions
@app.route('/book_appointment', methods=['POST'])
def book_appointment():
    data = request.json
    appointment_date = data['appointmentDate']
    client_name = data['clientName']
    client_email = data['clientEmail']
    lawyer_name = data['lawyerName']
    appointment_time = data['appointmentTime']
    case_details = data['caseDetails']
      # Get the lawyer name from the request

    try:
        # Parse the date and time to a datetime object
        appointment_datetime = datetime.strptime(f"{appointment_date} {appointment_time}", "%Y-%m-%d %H:%M")
        appointment_end_time = appointment_datetime + timedelta(minutes=30)

        # Connect to database and check for overlapping appointments
        conn = connect_db()
        cur = conn.cursor()

        check_query = """
        SELECT * FROM public.clientappointments
        WHERE lawyer_name = %s 
        AND appointment_date = %s
        AND (
            (appointment_time <= %s AND appointment_time + interval '30 minutes' > %s)
            OR (appointment_time >= %s AND appointment_time < %s)
        );
        """

        cur.execute(check_query, (lawyer_name, appointment_date, appointment_datetime.time(), appointment_datetime.time(), appointment_datetime.time(), appointment_end_time.time()))
        overlapping_appointments = cur.fetchall()

        if overlapping_appointments:
            cur.close()
            conn.close()
            return jsonify({"message": f"This time slot is already booked for {lawyer_name}. Please choose another time."}), 400

        # Insert the new appointment into the database
        insert_query = """
        INSERT INTO public.clientappointments (appointment_date, client_name, client_email, lawyer_name, appointment_time, case_details)
        VALUES (%s, %s, %s, %s, %s, %s);
        """
        cur.execute(insert_query, (appointment_date, client_name, client_email, lawyer_name, appointment_time, case_details))
        conn.commit()

    # Send confirmation email
       # print(f"Loaded email: {app.config.get('MAIL_USERNAME')}")
       #print(f"Loaded password: {app.config.get('MAIL_PASSWORD')}")
       # Send confirmation email using the separate email service
        email_sent = send_email(client_name, client_email, appointment_date, appointment_time, case_details, lawyer_name)

        if email_sent:
            return jsonify({"message": "Appointment booked successfully, and confirmation email sent!"})
        else:
            return jsonify({"message": "Appointment booked, but email confirmation failed."}), 500

    finally:
        # Ensure the database connection is closed
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()


if __name__ == '__main__':
    app.run(debug=True)

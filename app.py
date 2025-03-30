import os
import re
import nltk
import pandas as pd
from firebase_admin import db
from flask import Flask, render_template, request, jsonify, g, session, redirect, url_for, flash
from flask_session import Session
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
from admin import admin_bp, init_mail, db
from check_env import init_mail, send_email 
from case_tracking import case_tracking_bp
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
app.secret_key = "supersecretkey"
init_mail(app) 

# ✅ Configure Flask Sessions
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'  # Can also use 'redis' or 'memcached' for better persistence
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
Session(app)  # ✅ Initialize Session

# ✅ Register Blueprints
app.register_blueprint(admin_bp)

# ✅ Register Blueprints
app.register_blueprint(case_tracking_bp)

@app.route('/admin_dashboard')
def admin_dashboard():
    if not session.get("admin_logged_in"):
        return render_template('admin_dashboard.html', admin_logged_in=False)
    return render_template('admin_dashboard.html', admin_logged_in=True)

# ✅ Check Admin Session Before Loading Dashboard
@app.route('/admin_check', methods=['GET'])
def admin_check():
    return jsonify({"logged_in": session.get("admin_logged_in", False)})

# In your main Flask app file (app.py or similar)
@app.route('/case_tracking')
def case_tracking():
    return render_template('case_tracking.html')


# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
def fetch_lawyer_profiles():
    ref = db.reference("lawyers_profile/lawyer_profile")  # Use reference, not collection
    lawyers = ref.get()  # Fetch all lawyers' data

    if lawyers:
        return list(lawyers.values())  # Convert dictionary values to a list
    else:
        return []
    


# Load a pre-trained model for text classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define practice areas
practice_areas = [
    'Corporate Lawyer', 'Civil Lawyer', 'Criminal Lawyer', 'Constitutional Lawyer',
    'Administrative Lawyer', 'Business Lawyer', 'Intellectual Property Lawyer',
    'Patent Lawyer', 'Trademark Lawyer', 'Copyright Lawyer', 'Environmental Lawyer',
    'Banking and Finance Lawyer', 'Bankruptcy Lawyer', 'Civil Rights Lawyer',
    'Family Lawyer', 'Employment Lawyer', 'Immigration Lawyer', 'Personal Injury Lawyer',
    'Tax Lawyer', 'Military Lawyer', 'International Lawyer', 'Municipal Lawyer',
    'Animal Lawyer', 'Education Lawyer', 'Elder Lawyer', 'Entertainment Lawyer',
    'Sports Lawyer', 'Securities Lawyer', 'Health Lawyer', 'Real Estate Lawyer',
    'Maritime Lawyer', 'Labor Lawyer'
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
            text = ''.join(page.extract_text() or '' for page in reader.pages)
            return text.strip()
        elif file.filename.endswith('.docx'):
            doc = docx.Document(file)
            text = '\n'.join([para.text for para in doc.paragraphs])
            return text.strip()
    except Exception as e:
        print(f"Error extracting text from file: {e}")
    return ''



@app.route('/recommend_lawyers', methods=['GET', 'POST'])
def recommend_lawyers_route():
    lawyer_recommendations = None
    error_message = None
    sort_order = None  

    if request.method == 'POST':
        user_query, min_price, max_price, location = None, None, None, None

        try:
            # Handle typed query
            if 'query' in request.form and request.form['query'].strip():
                user_query = request.form['query'].strip()
            
            # Handle document upload
            if 'upload' in request.files and request.files['upload'].filename:
                file = request.files['upload']
                
                # Ensure file is actually a file and not empty
                if file and file.filename.endswith(('.pdf', '.docx')):
                    document_text = extract_text_from_file(file)
                    if document_text:
                        user_query = document_text
                    else:
                        error_message = "Failed to extract text from the uploaded document."
            
            # Get the price range if provided
            min_price = request.form.get('min_price')
            max_price = request.form.get('max_price')

            # Get sorting order
            sort_order = request.form.get('sort_order')

            # Get location filter
            location = request.form.get('location')

            # Proceed only if user_query is valid
            if user_query:
                lawyer_recommendations = recommend_lawyers(user_query, min_price, max_price, sort_order, location)
            else:
                error_message = "No query provided. Please type a query or upload a valid document."

        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            print(error_message)

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
        # List of keyword phrases that should trigger redirection
        trigger_phrases = [
            "recommend lawyer", "find me a lawyer", "suggest a lawyer",
            "need a lawyer", "best lawyer for", "hire a lawyer", "find a lawyer", 
            "suggest lawyer"
        ]

        # Check if any of the trigger phrases appear **consecutively** in the query
        if any(phrase in user_question for phrase in trigger_phrases):
            if session.get("user_logged_in"):  # Check if user is logged in
                return jsonify({
                    "response": 'I can help you find a lawyer!<a href="/recommend_lawyers.html">Recommend</a>'
                })
            else:
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
        'firebase_app_id': os.getenv('FIREBASE_APP_ID'),
        'firebase_database_url': os.getenv('FIREBASE_DATABASE_URL')
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
    """Fetch lawyer details from Firebase and recommend based on query."""
    cleaned_query = preprocess_query(query)
    if not cleaned_query.strip():
        raise ValueError("The query is empty after preprocessing.")

    # Identify practice area using zero-shot classification
    classification = classifier(cleaned_query, practice_areas)
    recommended_practice_area = classification['labels'][0]

    # Fetch all lawyers from Firebase
    ref = db.reference("lawyers_profile/lawyer_profile")
    all_lawyers = ref.get() or {}  # Handle case where no data exists
    lawyer_list = list(all_lawyers.values())  # Convert dictionary values to a list

    # Filter by practice area
    recommendations = [
        lawyer for lawyer in lawyer_list if lawyer.get("Practice_area") == recommended_practice_area
    ]

    # Convert price to float and filter by price range
    if min_price is not None and max_price is not None:
        try:
            min_price, max_price = float(min_price), float(max_price)
            recommendations = [
                lawyer for lawyer in recommendations
                if min_price <= float(lawyer.get("Nominal_fees_per_hearing", 0)) <= max_price
            ]
        except ValueError:
            pass  # Ignore conversion errors

    # Filter by location
    if location:
        recommendations = [
            lawyer for lawyer in recommendations
            if location.lower() in lawyer.get("Location", "").lower()
        ]

    # Sort recommendations by price
    if sort_order == 'low_to_high':
        recommendations.sort(key=lambda x: float(x.get("Nominal_fees_per_hearing", 0)))
    elif sort_order == 'high_to_low':
        recommendations.sort(key=lambda x: float(x.get("Nominal_fees_per_hearing", 0)), reverse=True)

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
            return jsonify({"success": True, "message": "Appointment booked successfully, and confirmation email sent!"})
        else:
            return jsonify({"success": False, "message": "Appointment booked, but email confirmation failed."}), 500
    

    finally:
        # Ensure the database connection is closed
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()


if __name__ == '__main__':
    app.run(debug=True)

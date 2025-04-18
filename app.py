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
from firebase_admin import db

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
@app.route('/client')
def client():
    return render_template('client.html')

@app.route('/client/appointments')
def client_appointment_detail():
    return render_template('client_appointment_detail.html', **g.firebase_config)

@app.route('/lawyer/appointments')
def lawyer_appointment_detail():
    return render_template('lawyer_appointment_detail.html', **g.firebase_config)

@app.route('/api/get_appointments')
def get_appointments():
    user_type = request.args.get('type')  # 'client' or 'lawyer'
    email = request.args.get('email')     # client email or lawyer email

    if not user_type or not email:
        return jsonify({'error': 'Missing parameters'}), 400

    try:
        ref = db.reference('bookings')
        all_bookings = ref.get() or {}

        if user_type == 'client':
            sanitized_email = email.replace('.', ',')
            client_appointments = all_bookings.get(sanitized_email, {})
            return jsonify({'appointments': list(client_appointments.values())})

        elif user_type == 'lawyer':
            lawyer_appointments = []
            for client_email, appointments in all_bookings.items():
                for appointment_id, appointment in appointments.items():
                    if appointment.get('lawyer_name') == email:
                        lawyer_appointments.append(appointment)
            return jsonify({'appointments': lawyer_appointments})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
def sanitize_email(email):
    return email.replace('.', ',')

def fetch_firebase_appointments_by_email(email):
    sanitized_email = sanitize_email(email)
    ref = db.reference(f'bookings/{sanitized_email}')
    return ref.get() or {}

def fetch_firebase_appointments_by_lawyer_id(barcouncil_id):
    root = db.reference('bookings')
    all_data = root.get()
    results = []
    if all_data:
        for client_email, bookings in all_data.items():
            for _, appointment in bookings.items():
                if appointment.get("Bar_Council_ID") == barcouncil_id:
                    results.append(appointment)
    return results

@app.route('/api/request_reschedule', methods=['POST'])
def request_reschedule():
    data = request.json
    appointment_id = data.get('id')
    new_date = data.get('new_date')
    new_time = data.get('new_time')
    requester = data.get('requester')

    try:
        bookings_ref = db.reference('bookings')
        all_bookings = bookings_ref.get() or {}
        for email_key, appointments in all_bookings.items():
            for key, appointment in appointments.items():
                if appointment.get("id") == appointment_id:
                    updates = {}
                    if requester == "client":
                        updates[f"{email_key}/{key}/reschedule_request"] = {
                            "new_date": new_date,
                            "new_time": new_time,
                            "status": "pending_lawyer_confirmation"
                        }
                        updates[f"{email_key}/{key}/status"] = "reschedule_requested"
                    elif requester == "lawyer":
                        updates[f"{email_key}/{key}/reschedule_request"] = {
                            "new_date": new_date,
                            "new_time": new_time,
                            "status": "pending_client_confirmation"
                        }
                        updates[f"{email_key}/{key}/status"] = "reschedule_requested"
                    bookings_ref.update(updates)
                    return jsonify({"message": "Reschedule request submitted."})
        return jsonify({"error": "Appointment not found."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route('/api/request_delete', methods=['POST'])
def request_delete():
    data = request.json
    appointment_id = data.get('id')
    requester = data.get('requester')

    try:
        bookings_ref = db.reference('bookings')
        all_bookings = bookings_ref.get() or {}
        for email_key, appointments in all_bookings.items():
            for key, appointment in appointments.items():
                if appointment.get("id") == appointment_id:
                    updates = {
                        f"{email_key}/{key}/status": "delete_requested",
                        f"{email_key}/{key}/delete_by": requester
                    }
                    bookings_ref.update(updates)
                    return jsonify({"message": "Delete request submitted."})
        return jsonify({"error": "Appointment not found."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
  "Criminal Law","Corporate Law","Family Law",
  "Intellectual Property (IP) Law","Real Estate Law","Employment & Labor Law",
  "Banking & Finance Law","Tax Law","Environmental Law","Immigration Law",
  "Cyber Law","Personal Injury Law","Constitutional Law","Human Rights Law"
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
    location = None

    # Fetch unique locations from Firebase
    lawyers = fetch_lawyer_profiles()
    unique_locations = sorted(set(
        lawyer.get("Location") for lawyer in lawyers if lawyer.get("Location")
    ))

    if request.method == 'POST':
        try:
            # Handle typed query
            user_query = ""
            if 'query' in request.form and request.form['query'].strip():
                user_query = request.form['query'].strip()
            
            # Handle document upload
            if 'upload' in request.files and request.files['upload'].filename:
                file = request.files['upload']
                if file and file.filename.endswith(('.pdf', '.docx')):
                    document_text = extract_text_from_file(file)
                    if document_text:
                        user_query = document_text
                    else:
                        error_message = "Failed to extract text from the uploaded document."
            
            # Get the form data
            min_price = request.form.get('min_price')
            max_price = request.form.get('max_price')
            sort_order = request.form.get('sort_order')
            location = request.form.get('location')

            # Proceed only if we have a query
            if user_query:
                lawyer_recommendations = recommend_lawyers(
                    user_query, 
                    min_price, 
                    max_price, 
                    sort_order, 
                    location
                )
            elif not error_message:
                error_message = "Please provide a query or upload a document"

        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            print(error_message)

    return render_template(
        'recommend_lawyers.html',
        recommended_lawyers=lawyer_recommendations,
        error_message=error_message,
        unique_locations=unique_locations,
        selected_location=location,
        sort_order=sort_order
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
    all_lawyers = ref.get() or {}
    lawyer_list = list(all_lawyers.values())

    # Filter by practice area
    recommendations = [
        lawyer for lawyer in lawyer_list 
        if lawyer.get("Practice_area") == recommended_practice_area
    ]

    # Filter by price range
    if min_price is not None and max_price is not None:
        try:
            min_price = float(min_price)
            max_price = float(max_price)
            recommendations = [
                lawyer for lawyer in recommendations
                if min_price <= float(lawyer.get("Nominal_fees_per_hearing", 0)) <= max_price
            ]
        except ValueError:
            pass  # Ignore invalid price inputs

    # Filter by location
    if location:
        recommendations = [
            lawyer for lawyer in recommendations
            if location.lower() == lawyer.get("Location", "").lower()
        ]

    # Sort recommendations
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
        lawyer_name VARCHAR(100) NOT NULL,
        barcouncil_id VARCHAR(100) NOT NULL
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
    barcouncil_id = request.args.get('barcouncil_id', '').replace('/', '_')
    return render_template('booking.html', lawyer_name=lawyer_name, barcouncil_id=barcouncil_id)

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
    barcouncil_id = data.get('barcouncilid', '')
    
    print(f"[Server] Received Bar Council ID in request: {barcouncil_id}")

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

        cur.execute(check_query, (lawyer_name, appointment_date, appointment_datetime.time(), 
                                appointment_datetime.time(), appointment_datetime.time(), 
                                appointment_end_time.time()))
        overlapping_appointments = cur.fetchall()

        if overlapping_appointments:
            cur.close()
            conn.close()
            return jsonify({"message": f"This time slot is already booked for {lawyer_name}. Please choose another time."}), 400

        # Insert the new appointment into PostgreSQL (without barcouncilid)
        insert_query = """
        INSERT INTO public.clientappointments (
            appointment_date, client_name, client_email, 
            lawyer_name, appointment_time, case_details,barcouncil_id
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id;
        """
        cur.execute(insert_query, (
            appointment_date, client_name, client_email, 
            lawyer_name, appointment_time, case_details, barcouncil_id
        ))
        
        # Get the newly created appointment ID
        appointment_id = cur.fetchone()[0]
        conn.commit()

        # Send confirmation email
        email_sent = send_email(client_name, client_email, appointment_date, 
                              appointment_time, case_details, lawyer_name)
        
        # Push to Firebase with barcouncilid
        try:
            booking_data = {
                "id": appointment_id,
                "appointment_date": appointment_date,
                "client_name": client_name,
                "client_email": client_email,
                "appointment_time": appointment_time,
                "case_details": case_details,
                "lawyer_name": lawyer_name,
                "Bar_Council_ID": barcouncil_id  # Stored only in Firebase
            }
            # Push to Firebase (auto-generates unique key)
            sanitized_email = client_email.replace('.', ',')
            db.reference(f'bookings/{sanitized_email}').push().set(booking_data)
            
        except Exception as e:
            print(f"Firebase sync error: {e} (booking still saved to PostgreSQL)")
        
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
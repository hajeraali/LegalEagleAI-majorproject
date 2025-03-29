import os
import firebase_admin
from firebase_admin import credentials, db, auth
from flask import Blueprint, request, jsonify, session, url_for, redirect, render_template
from flask_mailman import Mail, EmailMessage
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
from contextlib import contextmanager
import logging

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

admin_bp = Blueprint('admin', __name__)

# Firebase Initialization
cred = credentials.Certificate("config/firebase_admin_config.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': os.getenv('FIREBASE_DB_URL')
})

# Flask-Mailman Configuration
mail = Mail()

def init_mail(app):
    app.config['MAIL_SERVER'] = 'smtp.gmail.com'
    app.config['MAIL_PORT'] = 587
    app.config['MAIL_USE_TLS'] = True
    app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
    app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
    app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_USERNAME')
    mail.init_app(app)

# Database connection helper
@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = psycopg2.connect(
            host="localhost",
            dbname="mydatabase",
            user="postgres",
            password="123"
        )
        yield conn
    except psycopg2.Error as e:
        logger.error(f"Database connection error: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

@contextmanager
def get_db_cursor():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except:
            conn.rollback()
            raise
        finally:
            cursor.close()

# Fixed Admin Credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# Check Admin Session
@admin_bp.route('/admin_check', methods=['GET'])
def admin_check():
    return jsonify({"logged_in": session.get("admin_logged_in", False)})

# Admin Login
@admin_bp.route('/admin_login', methods=['POST'])
def admin_login():
    data = request.json
    if data.get("username") == ADMIN_USERNAME and data.get("password") == ADMIN_PASSWORD:
        session["admin_logged_in"] = True
        return jsonify({"message": "Admin login successful"})
    return jsonify({"error": "Invalid credentials"}), 401

# Admin Logout
@admin_bp.route('/admin_logout', methods=['POST'])
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect(url_for('index'))

# Fetch Registered Users & Lawyers
@admin_bp.route('/registered_users', methods=['GET'])
def registered_users():
    if not session.get("admin_logged_in"):
        return jsonify({"error": "Unauthorized"}), 403

    try:
        users = db.reference('users').get() or {}
        lawyers = db.reference('lawyers').get() or {}

        for lawyer_id, lawyer_data in lawyers.items():
            lawyer_data["status"] = True if lawyer_data.get("status") else False

        return jsonify({"users": users, "lawyers": lawyers})

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return jsonify({"error": f"Error loading data: {str(e)}"}), 500

# Approve Lawyer
@admin_bp.route('/approve_lawyer', methods=['POST'])
def approve_lawyer():
    if not session.get("admin_logged_in"):
        return jsonify({"error": "Unauthorized"}), 403
    
    data = request.json
    lawyer_id = data.get("lawyer_id")

    if not lawyer_id:
        return jsonify({"error": "Lawyer ID is required"}), 400

    ref = db.reference(f'lawyers/{lawyer_id}')
    lawyer_data = ref.get()

    if not lawyer_data:
        return jsonify({"error": "Lawyer not found"}), 404

    ref.update({"status": True})

    send_email(
        subject="Account Approved",
        recipient=lawyer_data.get("email"),
        message=f"Hello {lawyer_data.get('name')},\n\nYour account has been approved."
    )

    return jsonify({"message": f"Lawyer {lawyer_data.get('name')} approved successfully!"})

# Reject Lawyer
@admin_bp.route('/reject_lawyer', methods=['POST'])
def reject_lawyer():
    if not session.get("admin_logged_in"):
        return jsonify({"error": "Unauthorized"}), 403
    
    data = request.json
    lawyer_id = data.get("lawyer_id")

    if not lawyer_id:
        return jsonify({"error": "Lawyer ID is required"}), 400

    ref = db.reference(f'lawyers/{lawyer_id}')
    lawyer_data = ref.get()

    if not lawyer_data:
        return jsonify({"error": "Lawyer not found"}), 404

    ref.update({"status": False})

    send_email(
        subject="Account Rejected",
        recipient=lawyer_data.get("email"),
        message=f"Hello {lawyer_data.get('name')},\n\nYour account has been rejected."
    )

    return jsonify({"message": f"Lawyer {lawyer_data.get('name')} rejected successfully!"})

# Send Email Function
def send_email(subject, recipient, message):
    try:
        email = EmailMessage(subject, message, os.getenv('MAIL_USERNAME'), [recipient])
        email.send()
        logger.info(f"Email sent to {recipient}")
    except Exception as e:
        logger.error(f"Email sending failed: {e}")

# Remove User (User/Lawyer)
@admin_bp.route('/remove_user', methods=['POST'])
def remove_user():
    if not session.get("admin_logged_in"):
        return jsonify({"error": "Unauthorized"}), 403

    data = request.json
    user_id = data.get("user_id")
    user_type = data.get("user_type")

    if not user_id or user_type not in ["lawyer", "user"]:
        return jsonify({"error": "Invalid request"}), 400

    ref = db.reference(f"{user_type}s/{user_id}")
    user_data = ref.get()

    if not user_data:
        return jsonify({"error": f"{user_type.capitalize()} not found"}), 404

    user_email = user_data.get("email")

    if not user_email:
        return jsonify({"error": "User email not found, cannot delete from Auth"}), 400

    try:
        user_record = auth.get_user_by_email(user_email)
        auth.delete_user(user_record.uid)
        logger.info(f"Deleted user from Firebase Auth: {user_record.uid}")
    except auth.UserNotFoundError:
        logger.warning(f"User not found in Firebase Auth, proceeding with DB deletion.")
    except Exception as e:
        return jsonify({"error": f"Failed to delete user from auth: {str(e)}"}), 500

    ref.delete()
    return jsonify({"message": f"{user_type.capitalize()} removed successfully from both DB and Auth!"})

# Update User Information
@admin_bp.route('/update_user', methods=['POST'])
def update_user():
    if not session.get("admin_logged_in"):
        return jsonify({"error": "Unauthorized"}), 403

    data = request.json
    user_id = data.get("user_id")
    user_type = data.get("user_type")
    update_data = data.get("update_data", {})

    if not user_id or user_type not in ["lawyer", "user"] or not update_data:
        return jsonify({"error": "Invalid request"}), 400

    ref = db.reference(f"{user_type}s/{user_id}")
    user_data = ref.get()

    if not user_data:
        return jsonify({"error": f"{user_type.capitalize()} not found"}), 404

    ref.update(update_data)
    return jsonify({"message": f"{user_type.capitalize()} updated successfully!"})

# Template routes
@admin_bp.route('/admin/lawyer_profile/<lawyer_id>')
def lawyer_profile_page(lawyer_id):
    if not session.get("admin_logged_in"):
        return redirect(url_for('admin.admin_login'))
    return render_template('lawyer_profile.html')

@admin_bp.route('/admin/client_appointments/<client_email>')
def client_appointments_page(client_email):
    if not session.get("admin_logged_in"):
        return redirect(url_for('admin.admin_login'))
    return render_template('client_appointments.html')

# API endpoints
@admin_bp.route('/api/lawyer_profile/<lawyer_id>')
def get_lawyer_profile(lawyer_id):
    if not session.get("admin_logged_in"):
        return jsonify({"error": "Unauthorized"}), 403
    
    try:
        lawyer_ref = db.reference(f'lawyers/{lawyer_id}')
        lawyer_basic = lawyer_ref.get()
        
        if not lawyer_basic:
            return jsonify({"error": "Lawyer not found"}), 404
        
        bar_council_id = lawyer_basic.get('barCouncilID')
        if not bar_council_id:
            return jsonify({"error": "Bar Council ID not found"}), 404
        
        encoded_bcid = bar_council_id.replace('/', '_')
        profile_ref = db.reference(f'lawyers_profile/lawyer_profile/{encoded_bcid}')
        lawyer_profile = profile_ref.get() or {}
        
        profile_data = {
            **lawyer_basic,
            **lawyer_profile
        }
        
        return jsonify({"profile": profile_data})
    except Exception as e:
        logger.error(f"Error getting lawyer profile: {str(e)}")
        return jsonify({"error": str(e)}), 500

@admin_bp.route('/api/lawyer_appointments/<lawyer_id>')
def get_lawyer_appointments(lawyer_id):
    if not session.get("admin_logged_in"):
        return jsonify({"error": "Unauthorized"}), 403
    
    try:
        lawyer_ref = db.reference(f'lawyers/{lawyer_id}')
        lawyer_data = lawyer_ref.get()
        if not lawyer_data:
            return jsonify({"error": "Lawyer not found"}), 404
        
        lawyer_name = lawyer_data.get('name')
        if not lawyer_name:
            return jsonify({"error": "Lawyer name not found"}), 404
        
        search_term = request.args.get('search', '')
        search_pattern = f"%{search_term}%"
        
        with get_db_cursor() as cur:
            query = """
                SELECT 
                    id, appointment_date, client_name, client_email,
                    appointment_time, case_details, lawyer_name
                FROM clientappointments 
                WHERE lawyer_name = %s
                AND (client_name ILIKE %s OR client_email ILIKE %s OR case_details ILIKE %s)
                ORDER BY appointment_date DESC, appointment_time DESC
            """
            cur.execute(query, (lawyer_name, search_pattern, search_pattern, search_pattern))
            
            appointments = []
            for row in cur.fetchall():
                appointments.append({
                    "id": row[0],
                    "appointment_date": row[1].strftime('%Y-%m-%d'),
                    "client_name": row[2],
                    "client_email": row[3],
                    "appointment_time": str(row[4]),
                    "case_details": row[5],
                    "lawyer_name": row[6]
                })
            
        return jsonify({"appointments": appointments})
        
    except psycopg2.Error as e:
        logger.error(f"Database error: {str(e)}")
        return jsonify({"error": "Database operation failed"}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500

@admin_bp.route('/api/client_info/<client_email>')
def get_client_info(client_email):
    if not session.get("admin_logged_in"):
        return jsonify({"error": "Unauthorized"}), 403
    
    try:
        users_ref = db.reference('users')
        users = users_ref.get() or {}
        
        client = None
        for user_id, user_data in users.items():
            if user_data.get('email') == client_email:
                client = user_data
                break
                
        if not client:
            return jsonify({"error": "Client not found"}), 404
            
        return jsonify({"client": client})
    except Exception as e:
        logger.error(f"Error getting client info: {str(e)}")
        return jsonify({"error": str(e)}), 500

@admin_bp.route('/api/client_appointments/<client_email>')
def get_client_appointments(client_email):
    if not session.get("admin_logged_in"):
        return jsonify({"error": "Unauthorized"}), 403
    
    try:
        search_term = request.args.get('search', '')
        search_pattern = f"%{search_term}%"
        
        with get_db_cursor() as cur:
            query = """
                SELECT 
                    id, appointment_date, client_name, client_email,
                    appointment_time, case_details, lawyer_name
                FROM clientappointments 
                WHERE client_email = %s
                AND (client_name ILIKE %s OR case_details ILIKE %s OR lawyer_name ILIKE %s)
                ORDER BY appointment_date DESC, appointment_time DESC
            """
            cur.execute(query, (client_email, search_pattern, search_pattern, search_pattern))
            
            appointments = []
            for row in cur.fetchall():
                lawyer_name = row[6]
                lawyer_details = get_lawyer_details_by_name(lawyer_name)
                
                appointments.append({
                    "id": row[0],
                    "appointment_date": row[1].strftime('%Y-%m-%d'),
                    "client_name": row[2],
                    "client_email": row[3],
                    "appointment_time": str(row[4]),
                    "case_details": row[5],
                    "lawyer_name": lawyer_name,
                    "lawyer_email": lawyer_details.get('email', 'N/A'),
                    "lawyer_contact": lawyer_details.get('contact', 'N/A'),
                    "lawyer_barcouncil": lawyer_details.get('barCouncilID', 'N/A'),
                    "lawyer_practice_area": lawyer_details.get('Practice_area', 'N/A')
                })
            
        return jsonify({"appointments": appointments})
        
    except psycopg2.Error as e:
        logger.error(f"Database error: {str(e)}")
        return jsonify({"error": "Database operation failed"}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500

def get_lawyer_details_by_name(lawyer_name):
    """Helper function to get lawyer details from Firebase by name"""
    try:
        lawyers_ref = db.reference('lawyers')
        lawyers = lawyers_ref.get() or {}
        
        for lawyer_id, lawyer_data in lawyers.items():
            if lawyer_data.get('name', '').lower() == lawyer_name.lower():
                bar_council_id = lawyer_data.get('barCouncilID')
                if bar_council_id:
                    encoded_bcid = bar_council_id.replace('/', '_')
                    profile_ref = db.reference(f'lawyers_profile/lawyer_profile/{encoded_bcid}')
                    profile_data = profile_ref.get() or {}
                    return {**lawyer_data, **profile_data}
                return lawyer_data
        return {}
    except Exception as e:
        logger.error(f"Error fetching lawyer details: {str(e)}")
        return {}

@admin_bp.route('/api/delete_appointment', methods=['POST'])
def delete_appointment():
    if not session.get("admin_logged_in"):
        return jsonify({"error": "Unauthorized"}), 403

    data = request.json
    appointment_id = data.get("appointment_id")

    if not appointment_id:
        return jsonify({"error": "Appointment ID is required"}), 400

    try:
        with get_db_cursor() as cur:
            # First check if appointment exists
            cur.execute("SELECT id FROM clientappointments WHERE id = %s", (appointment_id,))
            if not cur.fetchone():
                return jsonify({"error": "Appointment not found"}), 404

            # Delete the appointment
            cur.execute("DELETE FROM clientappointments WHERE id = %s", (appointment_id,))
            
        return jsonify({"message": "Appointment deleted successfully"})
        
    except psycopg2.Error as e:
        logger.error(f"Database error deleting appointment: {str(e)}")
        return jsonify({"error": "Database operation failed"}), 500
    except Exception as e:
        logger.error(f"Unexpected error deleting appointment: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500

@admin_bp.route('/api/delete_lawyer_appointment', methods=['POST'])  # Changed endpoint name
def delete_lawyer_appointment():  # Changed function name
    if not session.get("admin_logged_in"):
        return jsonify({"error": "Unauthorized"}), 403

    data = request.json
    appointment_id = data.get("appointment_id")

    if not appointment_id:
        return jsonify({"error": "Appointment ID is required"}), 400

    try:
        with get_db_cursor() as cur:
            # First check if appointment exists
            cur.execute("SELECT id FROM clientappointments WHERE id = %s", (appointment_id,))
            if not cur.fetchone():
                return jsonify({"error": "Appointment not found"}), 404

            # Delete the appointment
            cur.execute("DELETE FROM clientappointments WHERE id = %s", (appointment_id,))
            
        return jsonify({"message": "Appointment deleted successfully"})
        
    except psycopg2.Error as e:
        logger.error(f"Database error deleting appointment: {str(e)}")
        return jsonify({"error": "Database operation failed"}), 500
    except Exception as e:
        logger.error(f"Unexpected error deleting appointment: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500
import os
import firebase_admin
from firebase_admin import credentials, db, auth
from flask import Blueprint, request, jsonify, session, url_for, redirect
from flask_mailman import Mail, EmailMessage
from dotenv import load_dotenv

load_dotenv()

admin_bp = Blueprint('admin', __name__)

# Firebase Initialization
cred = credentials.Certificate("config/firebase_admin_config.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': os.getenv('FIREBASE_DB_URL')
})

# Flask-Mailman Configuration
mail = Mail()

def init_mail(app):
    app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # Update if using another provider
    app.config['MAIL_PORT'] = 587
    app.config['MAIL_USE_TLS'] = True
    app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')  # Your email
    app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')  # Your email app password
    app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_USERNAME')
    mail.init_app(app)

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

        # Fix lawyer status display
        for lawyer_id, lawyer_data in lawyers.items():
            lawyer_data["status"] = True if lawyer_data.get("status") else False

        return jsonify({"users": users, "lawyers": lawyers})

    except Exception as e:
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

    # Update status to True (Approved)
    ref.update({"status": True})

    # Send approval email
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

    # Update status to False (Rejected)
    ref.update({"status": False})

    # Send rejection email
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
        print(f"✅ Email sent to {recipient}")
    except Exception as e:
        print(f"❌ Email sending failed: {e}")

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

    # Get the associated email from Firebase DB
    user_email = user_data.get("email")

    if not user_email:
        return jsonify({"error": "User email not found, cannot delete from Auth"}), 400

    # Find the Firebase Auth UID based on email
    try:
        user_record = auth.get_user_by_email(user_email)
        auth.delete_user(user_record.uid)  # Delete user from Firebase Authentication
        print(f"✅ Deleted user from Firebase Auth: {user_record.uid}")
    except auth.UserNotFoundError:
        print(f"⚠️ User not found in Firebase Auth, proceeding with DB deletion.")
    except Exception as e:
        return jsonify({"error": f"Failed to delete user from auth: {str(e)}"}), 500


    # Delete from Firebase Database
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

    # Update user data
    ref.update(update_data)
    return jsonify({"message": f"{user_type.capitalize()} updated successfully!"})
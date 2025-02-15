import os
from flask_mailman import Mail, EmailMessage
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

mail = Mail()  # Define Mail object globally

def init_mail(app):
    """Initialize Flask-Mailman with app configuration"""
    app.config["MAIL_SERVER"] = "smtp.gmail.com"
    app.config["MAIL_PORT"] = 587
    app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME")
    app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD")
    app.config["MAIL_USE_TLS"] = True
    app.config["MAIL_USE_SSL"] = False

    print(f"Loaded email: {app.config['MAIL_USERNAME']}")  # Debugging
    print(f"Loaded password: {'Yes' if app.config['MAIL_PASSWORD'] else 'No'}")  # Debugging

    mail.init_app(app)  # Initialize Flask-Mailman with the app

def send_email(client_name, client_email, appointment_date, appointment_time, case_details, lawyer_name):
    """Send an email confirmation using Flask-Mailman"""
    try:
        email_body = f"""Dear {client_name},

Your appointment has been successfully booked for {appointment_date} at {appointment_time} with {lawyer_name}.

Case Details:
{case_details}

Thank you for choosing our services.
"""

        msg = EmailMessage(
            subject="Appointment Confirmation",
            body=email_body,
            from_email=os.getenv("MAIL_USERNAME"),
            to=[client_email],
        )

        with mail.get_connection() as connection:
            connection.send_messages([msg])  # Send the email

        print("Email sent successfully!")  # Debug log
        return True

    except Exception as e:
        print(f"Failed to send email: {e}")  # Debug log
        return False

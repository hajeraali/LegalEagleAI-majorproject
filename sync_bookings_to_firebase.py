import os
import time
import psycopg2
import firebase_admin
from firebase_admin import credentials, db
from dotenv import load_dotenv
from datetime import datetime

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Firebase initialization
def initialize_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate("config/firebase_admin_config.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': os.getenv('FIREBASE_DB_URL')
        })
        logger.info("Firebase initialized successfully")

# PostgreSQL connection
def get_postgres_connection():
    try:
        return psycopg2.connect(
            host="localhost",
            dbname="mydatabase",
            user="postgres",
            password="123"
        )
    except psycopg2.Error as e:
        logger.error(f"PostgreSQL connection error: {e}")
        raise

# Remove duplicate bookings with same ID
def remove_redundant_entries():
    firebase_ref = db.reference('bookings')
    all_clients = firebase_ref.get() or {}

    for client_email, appointments in all_clients.items():
        if not isinstance(appointments, dict):
            logger.warning(f"Skipping corrupted data under {client_email}")
            continue

        seen_ids = {}
        for key, appt in appointments.items():
            if not isinstance(appt, dict):
                continue
            appt_id = appt.get('id')
            if not appt_id:
                continue
            if appt_id in seen_ids:
                logger.info(f"Removing duplicate booking ID {appt_id} under {client_email}")
                firebase_ref.child(client_email).child(key).delete()
            else:
                seen_ids[appt_id] = key

# Sync all PostgreSQL bookings to Firebase
def sync_bookings():
    initialize_firebase()
    firebase_ref = db.reference('bookings')

    # Optional: Clean duplicates first
    remove_redundant_entries()

    while True:
        try:
            conn = get_postgres_connection()
            cursor = conn.cursor()

            # No time filtering – sync everything
            query = """
            SELECT id, appointment_date, client_name, client_email, 
                   appointment_time, case_details, lawyer_name, barcouncil_id
            FROM clientappointments
            ORDER BY appointment_date, appointment_time
            """
            cursor.execute(query)
            all_bookings = cursor.fetchall()

            if all_bookings:
                logger.info(f"Found {len(all_bookings)} bookings to sync")

                for booking in all_bookings:
                    booking_id, appointment_date, client_name, client_email, \
                    appointment_time, case_details, lawyer_name, barcouncil_id = booking

                    booking_data = {
                        "id": booking_id,
                        "appointment_date": appointment_date.strftime('%Y-%m-%d'),
                        "client_name": client_name,
                        "client_email": client_email,
                        "appointment_time": str(appointment_time),
                        "case_details": case_details,
                        "lawyer_name": lawyer_name,
                        "Bar_Council_ID": barcouncil_id or ""
                    }

                    try:
                        sanitized_email = client_email.replace('.', ',')
                        client_ref = firebase_ref.child(sanitized_email)
                        existing_bookings = client_ref.get() or {}
                        booking_exists = False

                        for key, existing in existing_bookings.items():
                            if isinstance(existing, dict) and existing.get('id') == booking_id:
                                # Always update Bar_Council_ID if missing or outdated
                                if existing.get("Bar_Council_ID") != (barcouncil_id or ""):
                                    client_ref.child(key).update({
                                        "Bar_Council_ID": barcouncil_id or ""
                                    })
                                    logger.info(f"Updated Bar_Council_ID for booking {booking_id}")
                                booking_exists = True
                                break

                        if not booking_exists:
                            client_ref.push().set(booking_data)
                            logger.info(f"Synced new booking {booking_id} for {client_email}")

                    except Exception as e:
                        logger.error(f"Failed to sync booking {booking_id}: {e}")

            else:
                logger.info("No bookings found to sync")

            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Error during sync: {e}")
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()

        time.sleep(60)

if __name__ == "__main__":
    sync_bookings()
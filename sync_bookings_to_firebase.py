# sync_bookings_to_firebase.py
import os
import time
import psycopg2
import firebase_admin
from firebase_admin import credentials, db
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Firebase initialization
def initialize_firebase():
    cred = credentials.Certificate("config/firebase_admin_config.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': os.getenv('FIREBASE_DB_URL')
    })
    logger.info("Firebase initialized successfully")

# PostgreSQL connection
def get_postgres_connection():
    try:
        conn = psycopg2.connect(
            host="localhost",
            dbname="mydatabase",
            user="postgres",
            password="123"
        )
        return conn
    except psycopg2.Error as e:
        logger.error(f"PostgreSQL connection error: {e}")
        raise

# Get the last sync timestamp
def get_last_sync_time(firebase_ref):
    try:
        last_sync = firebase_ref.child("_last_sync").get()
        return datetime.fromisoformat(last_sync) if last_sync else None
    except Exception as e:
        logger.error(f"Error getting last sync time: {e}")
        return None

# Set the last sync timestamp
def set_last_sync_time(firebase_ref, sync_time):
    try:
        firebase_ref.child("_last_sync").set(sync_time.isoformat())
    except Exception as e:
        logger.error(f"Error setting last sync time: {e}")

# Sync new bookings to Firebase
def sync_bookings():
    initialize_firebase()
    firebase_ref = db.reference('bookings')
    
    while True:
        try:
            conn = get_postgres_connection()
            cursor = conn.cursor()
            
            # Get the last sync time from Firebase
            last_sync = get_last_sync_time(firebase_ref)
            
            # Query to get new bookings since last sync
            if last_sync:
                query = """
                SELECT id, appointment_date, client_name, client_email, 
                       appointment_time, case_details, lawyer_name
                FROM clientappointments
                WHERE (appointment_date || ' ' || appointment_time)::timestamp > %s
                ORDER BY appointment_date, appointment_time
                """
                cursor.execute(query, (last_sync,))
            else:
                # First time sync - get all bookings
                query = """
                SELECT id, appointment_date, client_name, client_email, 
                       appointment_time, case_details, lawyer_name
                FROM clientappointments
                ORDER BY appointment_date, appointment_time
                """
                cursor.execute(query)
            
            new_bookings = cursor.fetchall()
            
            if new_bookings:
                logger.info(f"Found {len(new_bookings)} new bookings to sync")
                
                # Process each new booking
                for booking in new_bookings:
                    booking_id, appointment_date, client_name, client_email, \
                    appointment_time, case_details, lawyer_name = booking
                    
                    # Create booking data structure
                    booking_data = {
                        "id": booking_id,
                        "appointment_date": appointment_date.strftime('%Y-%m-%d'),
                        "client_name": client_name,
                        "client_email": client_email,
                        "appointment_time": str(appointment_time),
                        "case_details": case_details,
                        "lawyer_name": lawyer_name,
                        "synced_at": datetime.now().isoformat()
                    }
                    
                    # Push to Firebase under client_email
                    try:
                        # Create a new child under client_email with a unique key
                        client_ref = firebase_ref.child(client_email.replace('.', ','))  # Replace . with , for Firebase key
                        new_booking_ref = client_ref.push()
                        new_booking_ref.set(booking_data)
                        logger.info(f"Synced booking {booking_id} for client {client_email}")
                    except Exception as e:
                        logger.error(f"Failed to sync booking {booking_id} to Firebase: {e}")
                
                # Update last sync time to the latest booking's time
                latest_booking = new_bookings[-1]
                latest_timestamp = datetime.combine(latest_booking[1], latest_booking[4])
                set_last_sync_time(firebase_ref, latest_timestamp)
            else:
                logger.info("No new bookings found to sync")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error during sync: {e}")
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
        
        # Wait for 60 seconds before checking again
        time.sleep(60)

if __name__ == "__main__":
    sync_bookings()
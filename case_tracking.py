import psycopg2
from flask import Blueprint, request, jsonify, session
from datetime import datetime
from check_env import send_email

case_tracking_bp = Blueprint('case_tracking', __name__)

# ✅ Database Connection
def connect_db():
    return psycopg2.connect(
        host="localhost",
        dbname="mydatabase",
        user="postgres",
        password="123"
    )

# ✅ Fetch All Cases for a Client (Handles Multiple Appointments)
@case_tracking_bp.route('/case/client/<client_email>', methods=['GET'])
def view_client_cases(client_email):
    user_role = get_user_role()
    if not user_role:
        return jsonify({"error": "Unauthorized"}), 403

    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM case_tracking 
        WHERE client_email = %s
        ORDER BY appointment_date DESC;
    """, (client_email,))

    cases = cur.fetchall()
    conn.close()

    if not cases:
        return jsonify({"error": "No cases found"}), 404

    case_keys = ["id", "client_email", "lawyer_name", "appointment_date", "appointment_time",
                 "case_details", "case_updates", "rating", "review", "fees_paid", "fees_pending", "status"]
    cases_list = [dict(zip(case_keys, case)) for case in cases]

    return jsonify({"cases": cases_list})

# ✅ Fetch All Cases for a Lawyer (Handles Multiple Clients)
@case_tracking_bp.route('/case/lawyer/<lawyer_name>', methods=['GET'])
def view_lawyer_cases(lawyer_name):
    if get_user_role() != "lawyer":
        return jsonify({"error": "Unauthorized"}), 403

    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM case_tracking 
        WHERE lawyer_name = %s
        ORDER BY appointment_date DESC;
    """, (lawyer_name,))

    cases = cur.fetchall()
    conn.close()

    if not cases:
        return jsonify({"error": "No cases found"}), 404

    case_keys = ["id", "client_email", "lawyer_name", "appointment_date", "appointment_time",
                 "case_details", "case_updates", "rating", "review", "fees_paid", "fees_pending", "status"]
    cases_list = [dict(zip(case_keys, case)) for case in cases]

    return jsonify({"cases": cases_list})

# ✅ Update Case Progress (Both Lawyer and Client Can Update)
@case_tracking_bp.route('/case/update', methods=['POST'])
def update_case():
    data = request.json
    case_id = data["case_id"]
    case_update = data["case_update"]

    conn = connect_db()
    cur = conn.cursor()

    cur.execute("""
        UPDATE case_tracking 
        SET case_updates = CONCAT(case_updates, %s)
        WHERE id = %s;
    """, (f"\n{datetime.now().strftime('%Y-%m-%d %H:%M')}: {case_update}", case_id))

    conn.commit()
    conn.close()
    
    return jsonify({"message": "Case updated successfully!"})

# ✅ Submit Review (Only Client)
@case_tracking_bp.route('/case/review', methods=['POST'])
def submit_review():
    if get_user_role() != "client":
        return jsonify({"error": "Unauthorized"}), 403

    data = request.json
    case_id = data["case_id"]
    rating = data["rating"]
    review = data["review"]

    if len(review.split()) < 50:
        return jsonify({"error": "Review must be at least 50 words"}), 400

    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        UPDATE case_tracking 
        SET rating = %s, review = %s
        WHERE id = %s;
    """, (rating, review, case_id))

    conn.commit()
    conn.close()
    
    return jsonify({"message": "Review submitted successfully!"})

# ✅ Get User Role
def get_user_role():
    if session.get("admin_logged_in"):
        return "admin"
    elif session.get("lawyer_logged_in"):
        return "lawyer"
    elif session.get("client_logged_in"):
        return "client"
    return None

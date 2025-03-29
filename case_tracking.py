import psycopg2
from flask import Blueprint, request, jsonify, session

case_tracking_bp = Blueprint('case_tracking', __name__)

def connect_db():
    return psycopg2.connect(
        host="localhost",
        dbname="mydatabase",
        user="postgres",
        password="123"
    )

@case_tracking_bp.route('/case/update_status', methods=['POST'])
def update_case_status():
    if not session.get("admin_logged_in"):
        return jsonify({"error": "Unauthorized"}), 403
    
    data = request.json
    case_id = data.get("case_id")
    new_status = data.get("status")
    
    if not case_id or not new_status:
        return jsonify({"error": "Missing case_id or status"}), 400
    
    conn = connect_db()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            UPDATE case_tracking
            SET status = %s
            WHERE id = %s
            RETURNING *;
        """, (new_status, case_id))
        
        updated_case = cur.fetchone()
        conn.commit()
        
        if not updated_case:
            return jsonify({"error": "Case not found"}), 404
            
        return jsonify({
            "message": "Case status updated successfully",
            "case": dict(zip(
                ["id", "client_email", "lawyer_name", "appointment_date", "appointment_time",
                 "case_details", "case_updates", "rating", "review", "fees_paid", "fees_pending", "status"],
                updated_case
            ))
        })
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        conn.close()

@case_tracking_bp.route('/case/lawyer_cases/<lawyer_name>', methods=['GET'])
def get_lawyer_cases(lawyer_name):
    if not session.get("admin_logged_in") and not session.get("lawyer_logged_in"):
        return jsonify({"error": "Unauthorized"}), 403
    
    conn = connect_db()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            SELECT * FROM case_tracking
            WHERE lawyer_name = %s
            ORDER BY appointment_date DESC;
        """, (lawyer_name,))
        
        cases = cur.fetchall()
        
        cases_list = []
        for case in cases:
            cases_list.append(dict(zip(
                ["id", "client_email", "lawyer_name", "appointment_date", "appointment_time",
                 "case_details", "case_updates", "rating", "review", "fees_paid", "fees_pending", "status"],
                case
            )))
            
        return jsonify({"cases": cases_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        conn.close()

@case_tracking_bp.route('/case/client_cases/<client_email>', methods=['GET'])
def get_client_cases(client_email):
    if (not session.get("admin_logged_in") and 
        not session.get("client_logged_in") and 
        session.get("client_email") != client_email):
        return jsonify({"error": "Unauthorized"}), 403
    
    conn = connect_db()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            SELECT * FROM case_tracking
            WHERE client_email = %s
            ORDER BY appointment_date DESC;
        """, (client_email,))
        
        cases = cur.fetchall()
        
        cases_list = []
        for case in cases:
            cases_list.append(dict(zip(
                ["id", "client_email", "lawyer_name", "appointment_date", "appointment_time",
                 "case_details", "case_updates", "rating", "review", "fees_paid", "fees_pending", "status"],
                case
            )))
            
        return jsonify({"cases": cases_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        conn.close()


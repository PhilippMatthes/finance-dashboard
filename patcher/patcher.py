
import os
import hashlib
import glob
import json

from flask import Flask, request, jsonify, redirect
import psycopg2

app = Flask(__name__)

POSTGRES_CONF = {
    'dbname': os.getenv("POSTGRES_DB", "postgres"),
    'user': os.getenv("POSTGRES_USER", "postgres"),
    'password': os.getenv("POSTGRES_PASSWORD"),
    'host': os.getenv("POSTGRES_HOST", "postgres"),
    'port': os.getenv("POSTGRES_PORT", "5432"),
}

DASHBOARD_BASE_URL = os.getenv(
    "DASHBOARD_BASE_URL",
    "http://localhost:3000/d/finance-dashboard/finance-dashboard",
)

@app.route('/transactions/internal/toggle', methods=['GET'])
def update_internal():
    # Fetch the value of the transaction first and then apply the opposite.
    # Only backup the second query to ensure consistent results when replaying.
    transaction_hash = request.args.get('hash')
    if not transaction_hash:
        return jsonify({"error": "Missing 'hash' parameter"}), 400

    fetch_query = "SELECT internal FROM transactions WHERE hash = %s"
    update_query = "UPDATE transactions SET internal = %s WHERE hash = %s"

    with psycopg2.connect(**POSTGRES_CONF) as connection, connection.cursor() as cursor:
        cursor.execute(fetch_query, (transaction_hash,))
        result = cursor.fetchone()
        if result is None:
            return jsonify({"error": "Transaction not found"}), 404

        current_internal = result[0]
        new_internal = not current_internal

        cursor.execute(update_query, (new_internal, transaction_hash))
        backup([update_query, (new_internal, transaction_hash)])

    return redirect(f"{DASHBOARD_BASE_URL}?update_internal=successful", code=302)

@app.route('/transactions/patch', methods=['GET'])
def patch():
    # Apply fields from the request to the transaction.
    transaction_hash = request.args.get('hash')
    if not transaction_hash:
        return jsonify({"error": "Missing 'hash' parameter"}), 400

    update_args = (
        request.args.get('primary_class'),
        request.args.get('secondary_class'),
    )
    update_query = """
    UPDATE transactions SET
        primary_class = %s,
        secondary_class = %s
    WHERE hash = %s
    """

    with psycopg2.connect(**POSTGRES_CONF) as connection, connection.cursor() as cursor:
        cursor.execute(update_query, (*update_args, transaction_hash))
        backup([update_query, (*update_args, transaction_hash)])

    return redirect(f"{DASHBOARD_BASE_URL}?var-patch=successful", code=302)

@app.route('/apply-patches', methods=['GET'])
def apply_patches():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "patches")
    for file in glob.glob(os.path.join(input_dir, "*.json")):
        with open(file) as f:
            q = json.loads(f.read())
        with psycopg2.connect(**POSTGRES_CONF) as connection, connection.cursor() as cursor:
            cursor.execute(*q)
    return redirect(f"{DASHBOARD_BASE_URL}?var-patch=successful", code=302)

def backup(queryargs):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "patches")
    h = hashlib.sha256()
    h.update(json.dumps(queryargs).encode())
    hash = h.hexdigest()
    with open(os.path.join(input_dir, f"{hash}.json"), 'w') as f:
        f.write(json.dumps(queryargs))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001)
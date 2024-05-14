import psycopg2
from flask import Flask


def create_app():
    app = Flask(__name__)

    @app.route("/customer/<string:customer_id>")
    def customer(customer_id):
        try:
            # Connect to your postgres DB
            conn = psycopg2.connect(
                "postgresql://postgres:postgres@localhost:15432/postgres?sslmode=disable"
            )

            # Open a cursor to perform database operations
            cur = conn.cursor()

            # Execute a query
            cur.execute(f"SELECT * FROM customer WHERE customer_id = {customer_id};")

            # Retrieve query results
            records = cur.fetchall()

            return records
        except Exception as e:
            return f"Error: {e}", 400

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(port=3000, debug=True)

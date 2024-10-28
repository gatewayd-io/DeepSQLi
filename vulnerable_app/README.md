# Vulnerable Customer App

This is a simple, intentionally vulnerable web application designed for testing SQL injection (SQLi) detection. The app is built with Flask, connects to a PostgreSQL database via GatewayD, and exposes a vulnerability in the `/customer/<customer_id>` endpoint that allows SQL injection through unsanitized user input.

> [!WARNING]
> This application is vulnerable to SQL injection and is for testing purposes only. Do not deploy this application in production or on a public server.

## Setup and Installation

```bash
git clone git@github.com:gatewayd-io/DeepSQLi.git
cd DeepSQLi/vulnerable_app
pip install -r requirements.txt
python main.py
```

The app will start in debug mode and listen on `http://localhost:3000`.

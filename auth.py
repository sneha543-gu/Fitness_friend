# auth.py

import csv
import os

USER_DB = "users.csv"

# Create users.csv with header if it doesn't exist
if not os.path.exists(USER_DB):
    with open(USER_DB, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["username", "password"])

# Register new user
def register_user(username, password):
    if user_exists(username):
        return False
    with open(USER_DB, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([username, password])
    return True

# Check if user already exists
def user_exists(username):
    with open(USER_DB, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["username"] == username:
                return True
    return False

# Authenticate user login
def authenticate_user(username, password):
    with open(USER_DB, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["username"] == username and row["password"] == password:
                return True
    return False

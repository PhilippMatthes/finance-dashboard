"""
JSON transactions importer.

Usage: Add a file in input/*.json that follows the format below. The importer
will read all json files in the input directory and return the transactions:

[
    {
        "iban": "DE00000000000000000000",
        "date": "01.01.2025",
        "internal": false,
        "client": "Company XYZ",
        "kind": "Credit",
        "purpose": "Salary",
        "amount": 1000,
        "balance": 10000,
        "currency": "EUR"
    }
]
"""

import os
import glob
import json
import datetime


def fetch_transactions():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "input")
    print(f"Reading JSON banking transactions from {input_dir}")
    for file in glob.glob(os.path.join(input_dir, "*.json")):
        with open(file) as f:
            transactions = json.loads(f.read())
    for transaction in transactions:
        # Parse date as string -> format: 01.01.1970
        transaction['date'] = datetime.datetime.strptime(transaction['date'], "%d.%m.%Y").date()
        yield transaction

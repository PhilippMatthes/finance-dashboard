import os
import psycopg2
import pandas as pd
import hashlib


POSTGRES_CONF = {
    'dbname': os.getenv("POSTGRES_DB", "postgres"),
    'user': os.getenv("POSTGRES_USER", "postgres"),
    'password': os.getenv("POSTGRES_PASSWORD"),
    'host': os.getenv("POSTGRES_HOST", "postgres"),
    'port': os.getenv("POSTGRES_PORT", "5432"),
}

# Check if the DB is up.
try:
    with psycopg2.connect(**POSTGRES_CONF) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
except psycopg2.OperationalError as e:
    print("Waiting for the DB to be ready...")
    exit(1)

# The plugins will be used to fetch bank transactions and return them in a
# format that can be processed by the syncer. The plugins should have a function
# called `fetch_transactions` that return a list of transactions.
IMPORTER_PLUGINS = os.getenv("IMPORTER_PLUGINS", None)
if IMPORTER_PLUGINS is None:
    print("No importer plugin specified.")
    exit(1)
# Separate multiple plugins by comma.
IMPORTER_PLUGINS = IMPORTER_PLUGINS.split(',')
# Try to import the plugin.
try:
    importers = [
        __import__(plugin, fromlist=['fetch_transactions'])
        for plugin in IMPORTER_PLUGINS
    ]
except ImportError as e:
    print(f"Failed to import plugin: {e}")
    exit(1)
print(f"Successfully imported plugins {IMPORTER_PLUGINS}")

# The plugin will be used to classify the bank transactions. The plugin should
# have a function called `classify_transactions` that takes a list of transactions
# and returns a list of classified transactions.
CLASSIFIER_PLUGIN = os.getenv("CLASSIFIER_PLUGIN", None)
if CLASSIFIER_PLUGIN is None:
    print("No classifier plugin specified.")
    exit(1)
# Try to import the plugin.
try:
    classifier = __import__(CLASSIFIER_PLUGIN, fromlist=['classify_transactions'])
except ImportError:
    print(f"Failed to import plugin: {CLASSIFIER_PLUGIN}")
    exit(1)
print(f"Successfully imported plugin: {CLASSIFIER_PLUGIN}")

# Create tables if they don't exist.
with psycopg2.connect(**POSTGRES_CONF) as connection, connection.cursor() as cursor:
    cursor.execute("""CREATE TABLE IF NOT EXISTS transactions (
        iban TEXT NOT NULL,
        internal BOOLEAN DEFAULT FALSE,
        date DATE NOT NULL,
        client TEXT NOT NULL,
        kind TEXT NOT NULL,
        purpose TEXT NOT NULL,
        amount DECIMAL NOT NULL,
        balance DECIMAL NOT NULL,
        currency TEXT NOT NULL,
        primary_class TEXT,
        secondary_class TEXT,
        hash VARCHAR(255) NOT NULL,
        PRIMARY KEY (iban, date, client, kind, purpose, amount, balance, currency)
    )""")

# Fetch transactions from all importers.
transactions = []
for importer in importers:
    transactions.extend(importer.fetch_transactions())

df = pd.DataFrame(transactions)

# Drop duplicate transactions to avoid normal transactions being marked as internal.
df.drop_duplicates(subset=['date', 'amount', 'client', 'purpose'], inplace=True)

# Strip all whitespace from ibans (e.g. " DE 1234 ..." -> "DE1234...")
strip = lambda x: x.replace(' ', '')
df['iban'] = df['iban'].apply(strip)

# Mark transactions seen on two accounts as internal in the column 'internal'.
# These aren't any expenses or income since they are between our own accounts.
df['amount_abs'] = df['amount'].abs()
# Only override None values in the internal column, in case the importer has
# already marked some transactions as internal.
df['internal'] = df.duplicated(subset=['date', 'amount_abs'], keep=False) | df['internal'].fillna(False)

# Classify transactions
df['primary_class'], df['secondary_class'] = zip(*df.apply(classifier.classify_transaction, axis=1))

# Calculate a hash for each transaction to create links in Grafana.
def sha256(t):
    h = hashlib.sha256()
    h.update(t['iban'].encode())
    h.update(str(t['date']).encode())
    h.update(t['client'].encode())
    h.update(t['kind'].encode())
    h.update(t['purpose'].encode())
    h.update(str(t['amount']).encode())
    h.update(str(t['balance']).encode())
    h.update(t['currency'].encode())
    return h.hexdigest()
df['hash'] = df.apply(sha256, axis=1)

# Insert transactions into the database.
with psycopg2.connect(**POSTGRES_CONF) as connection, connection.cursor() as cursor:
    for i, transaction in df.iterrows():
        cursor.execute("""
            INSERT INTO transactions (iban, internal, date, client, kind, purpose, amount, balance, currency, primary_class, secondary_class, hash)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (iban, date, client, kind, purpose, amount, balance, currency)
            DO NOTHING
        """, (
            transaction['iban'],
            transaction['internal'], # Whether the transaction is between our own accounts
            transaction['date'],
            transaction['client'],
            transaction['kind'],
            transaction['purpose'],
            transaction['amount'],
            transaction['balance'],
            transaction['currency'],
            transaction['primary_class'],
            transaction['secondary_class'],
            transaction['hash'],
        ))

print(f"Successfully inserted {i+1} transactions into the database.")

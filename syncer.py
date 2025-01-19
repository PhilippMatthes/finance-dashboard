import os
import psycopg2
import pandas as pd
import hashlib
import requests
import logging
import pydantic
import json

logger = logging.getLogger(__name__)

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
    logger.error("Waiting for the DB to be ready...")
    exit(1)

# Create tables if they don't exist.
with psycopg2.connect(**POSTGRES_CONF) as connection, connection.cursor() as cursor:
    cursor.execute("""CREATE TABLE IF NOT EXISTS transactions (
        hash VARCHAR(6) NOT NULL PRIMARY KEY,
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
        secondary_class TEXT
    )""")

OLLAMA_CONF = {
    "OLLAMA_HOST": os.getenv("OLLAMA_HOST", "ollama"),
    "OLLAMA_PORT": os.getenv("OLLAMA_PORT", "11434"),
    "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL", "mistral"),
}
# Check if ollama is up.
ollama_base_url = f"http://{OLLAMA_CONF['OLLAMA_HOST']}:{OLLAMA_CONF['OLLAMA_PORT']}"
ollama_model = OLLAMA_CONF['OLLAMA_MODEL']
try:
    requests.get(ollama_base_url)
except requests.exceptions.ConnectionError:
    logger.error("Waiting for ollama to be ready...")
    exit(1)
# Download the LLM that is used for classification.
logger.info(f"Downloading LLM: {OLLAMA_CONF['OLLAMA_MODEL']}")
model_response = requests.post(f"{ollama_base_url}/api/pull", json={
    "model": ollama_model, "stream": False,
})
if model_response.status_code != 200:
    logger.error(f"Failed to download LLM: {model_response.text}")
    exit(1)
logger.info(f"Successfully downloaded LLM: {OLLAMA_CONF['OLLAMA_MODEL']}")
logger.info(f"Loading LLM: {OLLAMA_CONF['OLLAMA_MODEL']}")
load_response = requests.post(f"{ollama_base_url}/api/generate", json={
    "model": ollama_model, "stream": False,
})
logger.info(f"Successfully loaded LLM: {OLLAMA_CONF['OLLAMA_MODEL']}")

# The plugins will be used to fetch bank transactions and return them in a
# format that can be processed by the syncer. The plugins should have a function
# called `fetch_transactions` that return a list of transactions.
IMPORTER_PLUGINS = os.getenv("IMPORTER_PLUGINS", None)
if IMPORTER_PLUGINS is None:
    logger.error("No importer plugin specified.")
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
    logger.error(f"Failed to import plugin: {e}")
    exit(1)
logger.info(f"Successfully imported plugins {IMPORTER_PLUGINS}")

# Fetch transactions from all importers.
transactions = []
for importer in importers:
    transactions.extend(importer.fetch_transactions())

df = pd.DataFrame(transactions)

# Calculate a hash for each transaction to make it identifiable.
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
    return h.hexdigest()[:6]
df['hash'] = df.apply(sha256, axis=1)

# Drop transactions that are already in the database.
with psycopg2.connect(**POSTGRES_CONF) as connection, connection.cursor() as cursor:
    cursor.execute("SELECT hash FROM transactions")
    hashes = set(h[0] for h in cursor.fetchall())
df = df[~df['hash'].isin(hashes)]

# Drop duplicate transactions to avoid normal transactions being marked as internal.
df.drop_duplicates(subset=['date', 'amount', 'client', 'purpose'], inplace=True)

# Mark transactions seen on two accounts as internal in the column 'internal'.
# These aren't any expenses or income since they are between our own accounts.
df['amount_abs'] = df['amount'].abs()
# Only override None values in the internal column, in case the importer has
# already marked some transactions as internal.
df['internal'] = df.duplicated(subset=['date', 'amount_abs'], keep=False) | df['internal'].fillna(False)

prompt = lambda csv: f"""
You are a bank transaction classifier. Classify the following transactions into
a primary and secondary class.

Examples for primary class:
"Vacation", "Sports", "Food", "Transport", "Health", "Shopping", "Income",
"Housing", "Utilities", "Entertainment", "Insurance", "Bank", "Donations",
"Taxes", "Gifts", "Books", ...

Examples for secondary class:
"Hotel X", "Person X", "Restaurant X", "Supermarket X", "Gym X", "Insurance X",
"Bank X", "Tax Office X", ...

If unclear, use "Other" as class.

Respond using JSON. Input(csv):

{csv}
"""

def classify(txns):
    """
    Call ollama LLM to generate a primary and secondary class for transactions.
    """
    class TransactionClassification(pydantic.BaseModel):
        client: str
        purpose: str
        primary_class: str
        secondary_class: str
    class TransactionClassificationList(pydantic.BaseModel):
        transactions: list[TransactionClassification]
    primary_classes, secondary_classes = [], []
    logger.info(f"Classifying transaction {i+1}/{len(transactions)}")
    csv = txns[['client', 'purpose']].to_csv(index=False, header=True)
    response = requests.post(f"{ollama_base_url}/api/generate", json={
        "model": ollama_model,
        "prompt": prompt(csv),
        "format": TransactionClassificationList.model_json_schema(),
        "stream": False,
        "options": {"num_ctx": 4096},
    })
    if response.status_code != 200:
        raise ValueError(response.text)
    logger.info(f"Successfully classified transactions: {response.text}")
    response_content = response.json()['response']
    lst = TransactionClassificationList.model_validate_json(response_content)
    primary_classes.extend([t.primary_class for t in lst])
    secondary_classes.extend([t.secondary_class for t in lst])
    return primary_classes, secondary_classes

# Insert transactions into the database.
slice_size = 20 # Should not be too large, otherwise the LLM context gets exceeded.
for i in range(0, len(transactions), slice_size):
    slice = df.iloc[i:i+slice_size]
    primary_classes, secondary_classes = classify(slice)
    for j, transaction in slice.iterrows():
        with psycopg2.connect(**POSTGRES_CONF) as connection, connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO transactions (hash, iban, internal, date, client, kind, purpose, amount, balance, currency, primary_class, secondary_class)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (hash)
                DO NOTHING
            """, (
                transaction['hash'],
                transaction['iban'],
                transaction['internal'], # Whether the transaction is between our own accounts
                transaction['date'],
                transaction['client'],
                transaction['kind'],
                transaction['purpose'],
                transaction['amount'],
                transaction['balance'],
                transaction['currency'],
                primary_classes[j],
                secondary_classes[j],
            ))
            logger.info(f"Successfully inserted transaction {transaction['hash']} into the database.")

logger.info(f"Successfully inserted {i+1} transactions into the database.")

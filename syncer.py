import os
import psycopg2
import pandas as pd
import hashlib
import requests
import pydantic

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
    print("Waiting for ollama to be ready...")
    exit(1)
# Download the LLM that is used for classification.
print(f"Downloading LLM: {OLLAMA_CONF['OLLAMA_MODEL']}")
model_response = requests.post(f"{ollama_base_url}/api/pull", json={
    "model": ollama_model, "stream": False,
})
if model_response.status_code != 200:
    print(f"Failed to download LLM: {model_response.text}")
    exit(1)
print(f"Successfully downloaded LLM: {OLLAMA_CONF['OLLAMA_MODEL']}")
print(f"Loading LLM: {OLLAMA_CONF['OLLAMA_MODEL']}")
load_response = requests.post(f"{ollama_base_url}/api/generate", json={
    "model": ollama_model, "stream": False,
})
print(f"Successfully loaded LLM: {OLLAMA_CONF['OLLAMA_MODEL']}")

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
Classify my bank transactions into primary and secondary classes.

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
        primary: str
        secondary: str
    class TransactionClassificationList(pydantic.BaseModel):
        transactions: list[TransactionClassification]
    csv = txns[['client']].to_csv(index=False, header=True)
    prompt_txt = prompt(csv)
    print(f"Prompting for classification: {prompt_txt}")
    response = requests.post(f"{ollama_base_url}/api/generate", json={
        "model": ollama_model,
        "prompt": prompt_txt,
        "format": TransactionClassificationList.model_json_schema(),
        "stream": False,
        "options": {"num_ctx": 4096},
    })
    if response.status_code != 200:
        raise ValueError(response.text)
    print(f"Successfully classified transactions: {response.text}")
    response_content = response.json()['response']
    lst = TransactionClassificationList.model_validate_json(response_content)
    return [t.primary for t in lst.transactions], [t.secondary for t in lst.transactions]

def insert(transaction, primary_class, secondary_class):
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
            primary_class,
            secondary_class,
        ))
        print(f"Successfully inserted transaction {transaction['hash']} into the database.")

# Make slices of transactions, otherwise the prompt will be too long for the LLM context.
slice_length = 50
for i in range(0, len(df), slice_length):
    slice = df.iloc[i:i+slice_length]
    primary_classes, secondary_classes = classify(slice)
    for j, (_, transaction) in enumerate(slice.iterrows()):
        insert(transaction, primary_classes[j], secondary_classes[j])

# Check if the transactions were inserted correctly.
with psycopg2.connect(**POSTGRES_CONF) as connection, connection.cursor() as cursor:
    cursor.execute("SELECT COUNT(*) FROM transactions")
    count = cursor.fetchone()[0]
print(f"Successfully inserted {count} transactions into the database.")

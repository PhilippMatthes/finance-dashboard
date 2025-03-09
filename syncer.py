import os
import psycopg2
import pandas as pd
import hashlib
import difflib


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
        description TEXT NOT NULL,
        amount DECIMAL NOT NULL,
        balance DECIMAL NOT NULL,
        currency TEXT NOT NULL,
        classification TEXT,
        rhythm TEXT,
        expected DATE,
        hash VARCHAR(255) NOT NULL,
        PRIMARY KEY (iban, date, description, amount, balance, currency)
    )""")

# Fetch transactions from all importers.
transactions = []
for importer in importers:
    transactions.extend(importer.fetch_transactions())

df = pd.DataFrame(transactions)

# Drop duplicate transactions to avoid normal transactions being marked as internal.
# Note: We make the duplicate removal solely based on the date and amount. This is
# because different importers may provide different encodings and metadata for the
# same transaction. E.g. when importing from CSV and PDF at the same time.
# The assumption here is that the same amount will only occur once on the same day.
# If you want to modify the behavior, set the TRANSACTION_DEDUPLICATION_COLUMNS env
# variable to a comma-separated list of columns to deduplicate on.
cols = os.getenv("TRANSACTION_DEDUPLICATION_COLUMNS", "date,amount").split(',')
df_wo_dupes = df.drop_duplicates(subset=cols, inplace=False)
# Print out which transactions were removed.
print(f"Removed {len(df) - len(df_wo_dupes)} duplicate transactions:")
print(df[~df.index.isin(df_wo_dupes.index)])
df = df_wo_dupes

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
df['classification'] = df.apply(classifier.classify_transaction, axis=1)

# Calculate a hash for each transaction to create links in Grafana.
def sha256(t):
    h = hashlib.sha256()
    h.update(t['iban'].encode())
    h.update(str(t['date']).encode())
    h.update(t['description'].encode())
    h.update(str(t['amount']).encode())
    h.update(str(t['balance']).encode())
    h.update(t['currency'].encode())
    return h.hexdigest()
df['hash'] = df.apply(sha256, axis=1)

# Find recurring transactions.
df['rhythm'] = None
df['expected'] = None
# Order by date (starting from the oldest transaction)
df = df.sort_values(by='date')
df['date_parsed'] = pd.to_datetime(df['date'])
print("Finding recurring transactions...")
for i, transaction in df.iterrows():
    if transaction['rhythm'] != None:
        # Already evaluated
        continue
    # Drop all for which we already found a rhythm (is not None)
    similar = df[df['rhythm'].isnull()]
    # Should have the same IBAN
    similar = similar[similar['iban'] == transaction['iban']]
    # Should have the same classification
    similar = similar[similar['classification'] == transaction['classification']]
    # Skip same transaction
    similar = similar[similar.index != i]

    # Should be within the same n% of the amount
    def similar_amount(t, n_pct_amount=0.2):
        if t['amount'] == 0 or transaction['amount'] == 0:
            return t['amount'] == transaction['amount']
        abs_diff = abs(t['amount'] - transaction['amount']) / abs(transaction['amount'])
        return abs_diff <= n_pct_amount
    similar = similar[similar.apply(similar_amount, axis=1)]

    # Should have a n% similar description
    def similar_description(t, n_pct_description=0.2):
        if not t['description'] or not transaction['description']:
            return t['description'] == transaction['description']
        seq = difflib.SequenceMatcher(None, t['description'].lower(), transaction['description'].lower())
        return seq.ratio() >= 1 - n_pct_description
    similar = similar[similar.apply(similar_description, axis=1)]

    # There should be a common rhythm in the transactions.
    # E.g. monthly, bi-monthly, quarterly, half-yearly, yearly
    rhythms = {
        'monthly': pd.DateOffset(months=1),
        'bi-monthly': pd.DateOffset(months=2),
        'quarterly': pd.DateOffset(months=3),
        'half-yearly': pd.DateOffset(months=6),
        'yearly': pd.DateOffset(years=1),
    }
    # For each rhythm, create a raster until today and check which raster fits the best.
    leeway = 3 # Number of businessdays before and after the transaction date to consider it a hit.
    # Take the last transaction date as the reference date.
    last_date = similar['date_parsed'].max()
    if last_date is pd.NaT:
        continue
    rasters = {
        rhythm: pd.concat([
            pd.Series(pd.date_range(
                transaction['date_parsed'] + pd.offsets.BusinessDay(n=delta) + freq,
                last_date + pd.offsets.BusinessDay(n=delta), freq=freq,
            ))
            for delta in range(-leeway, leeway + 1)
        ])
        for rhythm, freq in rhythms.items()
    }
    # Reward hits and penalize misses. If we don't penalize misses, monthly will always win.
    scores = {
        rhythm: (
            sum(raster.isin(similar['date_parsed'])) + sum(similar['date_parsed'].isin(raster)),
            (sum(~raster.isin(similar['date_parsed'])) / (leeway * 2)) + sum(~similar['date_parsed'].isin(raster)),
        ) # (hits, misses)
        for rhythm, raster in rasters.items()
    }
    def score(r):
        hits, misses = scores[r]
        if hits + misses == 0:
            return 0
        return hits / (hits + misses)
    best_rhythm = max(scores, key=score)

    similar = similar[similar['date_parsed'].isin(rasters[best_rhythm])]
    if len(similar) < 2:
        continue

    df.loc[similar.index, 'rhythm'] = best_rhythm
    # Calculate the expected date for the next transaction.
    expected = similar['date_parsed'] + rhythms[best_rhythm]
    # Make it database insertable.
    df.loc[similar.index, 'expected'] = expected.dt.strftime('%Y-%m-%d')

# Print the transactions with a rhythm.
print("Transactions with a rhythm:")
for i, transaction in df[df['rhythm'].notnull()].iterrows():
    print(transaction['date'], transaction['description'], transaction['rhythm'], transaction['expected'])

# Insert transactions into the database.
with psycopg2.connect(**POSTGRES_CONF) as connection, connection.cursor() as cursor:
    for i, transaction in df.iterrows():
        cursor.execute("""
            INSERT INTO transactions (iban, internal, date, description, amount, balance, currency, classification, rhythm, expected, hash)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (iban, date, description, amount, balance, currency)
            DO NOTHING
        """, (
            transaction['iban'],
            transaction['internal'], # Whether the transaction is between our own accounts
            transaction['date'],
            transaction['description'],
            transaction['amount'],
            transaction['balance'],
            transaction['currency'],
            transaction['classification'],
            transaction['rhythm'],
            transaction['expected'],
            transaction['hash'],
        ))

print(f"Successfully inserted {i+1} transactions into the database.")

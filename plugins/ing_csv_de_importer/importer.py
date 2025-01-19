"""
ING DE banking importer.

The importer reads all csv files in input/*.csv. These should be exported
from the ING DE banking website. The csv files are expected to contain a specific
header that contains meta information about the account. We will use this information
to distinguish between different accounts, allowing to dump any number of accounts
into the same input folder. Transactions between the given accounts will be dropped,
otherwise they will appear as income or expenses in the dashboard. This allows to
analyze the spending behavior of a single person, even if they have multiple accounts.

Header looks like this (encoded in latin1):

Umsatzanzeige;Datei erstellt am: 01.01.2025 00:00

IBAN;DE12 3456 7890 1234 5678 90
Kontoname;Girokonto
Bank;ING
Kunde;Max Mustermann
Zeitraum;01.01.2024 - 01.01.2025
Saldo;1234,56 EUR

Sortierung;Datum absteigend

In der CSV-Datei finden Sie alle bereits gebuchten Ums�tze. Die vorgemerkten Ums�tze werden nicht aufgenommen, auch wenn sie in Ihrem Internetbanking angezeigt werden.

Buchung;Valuta;Auftraggeber/Empf�nger;Buchungstext;Verwendungszweck;Saldo;W�hrung;Betrag;W�hrung
"""

import os
import glob
import pandas as pd

def parse_header(file):
    with open(file, 'r', encoding='latin1') as f:
        header = f.readline().strip()
        if "Umsatzanzeige;Datei erstellt am:" not in header:
            raise ValueError("Invalid banking csv header")
        f.readline() # Skip empty line
        iban = f.readline().split(';')[1].strip()
    return iban

def fetch_transactions():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "input")
    print(f"Reading ING DE banking transactions from {input_dir}")
    dfs = []
    for file in glob.glob(os.path.join(input_dir, "*.csv")):
        print(f"Reading file {file}")
        df = pd.read_csv(file, sep=';', encoding='latin1', skiprows=12)
        df['iban'] = parse_header(file) # Only need IBAN for deduplication
        # Fill NaNs in Auftraggeber/Empfaenger with "Unbekannt"
        df['client'] = df['Auftraggeber/Empfänger'].fillna('Unknown').astype(str)
        # Parse betrag as float -> format: 1.234,56 or -1.234,56
        df['amount'] = df['Betrag'].str.replace('.', '').str.replace(',', '.').astype(float)
        df['balance'] = df['Saldo'].str.replace('.', '').str.replace(',', '.').astype(float)
        # Buchung has format 14.10.2024, german
        df['date'] = pd.to_datetime(df['Buchung'], format='%d.%m.%Y')
        # Convert back to string for database insertion
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        df['kind'] = df['Buchungstext'].astype(str) # e.g. "Lastschrift", "Gutschrift"
        df['purpose'] = df['Verwendungszweck'].astype(str)
        df['currency'] = df['Währung.1'].astype(str)
        # If we have "Wertpapiergutschrift" or "Wertpapierkauf" in the purpose,
        # it's an internal transaction to/from the ING depot account.
        df['internal'] = df['kind'].str.contains("Wertpapiergutschrift|Wertpapierkauf")
        dfs.append(df)
    return pd.concat(dfs).to_dict(orient='records')
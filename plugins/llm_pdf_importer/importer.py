"""
Generic account statement importer using an LLM to extract transactions.
"""
import os
import glob
import pandas as pd
import PyPDF2
import datetime
import pydantic
import requests
import math

OLLAMA_CONF = {
    # Ollama needs to be running locally.
    "OLLAMA_URL": os.getenv("OLLAMA_URL", "http://host.docker.internal:11434"),
    "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL", "llama3.2"),
}
# Check if ollama is up.
ollama_base_url = OLLAMA_CONF['OLLAMA_URL']
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

class StatementMetadata(pydantic.BaseModel):
    iban: str
    start_balance: float
    end_balance: float

metadata_prompt_tpl = lambda input: f"""
### Instruction

Please provide the IBAN, start balance and end balance of the following
text that was extracted from an account statement.

Please respond using JSON format.

### Input

{input}
"""

class StatementTransaction(pydantic.BaseModel):
    date: datetime.date
    amount: float
    message: str
class StatementTransactionList(pydantic.BaseModel):
    transactions: list[StatementTransaction]

transaction_prompt_tpl = lambda input: f"""
### Instruction

Please provide the date, amount, and message of transactions in the following
text that was extracted from an account statement. The date must be the
date when the transaction was received.

Please respond using JSON format.

### Input

{input}
"""

def parse_account_statement_pdfs(pdfs):
    dfs = []
    for file in pdfs:
        print(f"Reading kontoauszug file {file}")
        # Open the PDF file
        reader = PyPDF2.PdfReader(file)
        pdf_text = ""
        for page in reader.pages:
            pdf_text += page.extract_text()

        metadata_prompt = metadata_prompt_tpl(pdf_text)
        print(f"Prompting LLM for bank statement metadata...")
        response = requests.post(f"{ollama_base_url}/api/generate", json={
            "model": ollama_model,
            "prompt": metadata_prompt,
            "format": StatementMetadata.model_json_schema(),
            "stream": False,
            # Modify this if the context length is unsupported by the model.
            "options": {"num_ctx": math.pow(2, 15)},
        })
        response.raise_for_status()
        metadata_obj = StatementMetadata.model_validate_json(response.json()["response"])
        print(f"Extracted metadata: {metadata_obj}")

        transaction_prompt = transaction_prompt_tpl(pdf_text)
        print(f"Prompting LLM for bank statement transactions...")
        response = requests.post(f"{ollama_base_url}/api/generate", json={
            "model": ollama_model,
            "prompt": transaction_prompt,
            "format": StatementTransactionList.model_json_schema(),
            "stream": False,
            # Modify this if the context length is unsupported by the model.
            "options": {"num_ctx": math.pow(2, 15)},
        })
        response.raise_for_status()
        transactions_obj = StatementTransactionList.model_validate_json(response.json()["response"])
        print(f"Extracted transactions: {transactions_obj}")

        # Convert to DataFrame
        df = pd.DataFrame([t.model_dump() for t in transactions_obj.transactions])
        df['iban'] = metadata_obj.iban
        # Convert 'date' column to datetime
        df['date'] = pd.to_datetime(df['date'])
        # Sum up the amounts to get the balance. Sort by date first.
        df = df.sort_values('date')
        df['balance'] = metadata_obj.start_balance + df['amount'].cumsum()
        # Check that the end balance is correct (.00 accuracy)
        diff = metadata_obj.end_balance - df['balance'].iloc[-1]
        if abs(diff) > 0.01:
            print(f"WARN End balance off by {diff}")
        # Convert back to string for database insertion
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        df['currency'] = 'EUR'
        dfs.append(df)
    return pd.concat(dfs)

def fetch_transactions():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "input")
    print(f"Reading banking transactions from {input_dir}")

    pdfs = glob.glob(os.path.join(input_dir, "*.pdf"))
    kontoauszug_df = parse_account_statement_pdfs([f for f in pdfs if "Kontoauszug" in f])

    all = pd.concat([kontoauszug_df])
    all = all.sort_values('date')
    return all.to_dict(orient='records')

if __name__ == "__main__":
    print(fetch_transactions())
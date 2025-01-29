import pandas as pd
import requests
import openai
from tqdm import tqdm
import os
from duckduckgo_search import DDGS
import time
import random
import hashlib
import pickle
import logging
from requests.exceptions import HTTPError

# ---------------------
# Configuration Section
# ---------------------

# Securely load API keys (use environment variables or config files)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

CACHE_FILE = 'search_cache.pkl'

logging.basicConfig(
    level=logging.INFO,
    filename='domain_search.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load or initialize cache
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'rb') as f:
        search_cache = pickle.load(f)
else:
    search_cache = {}

# ---------------------
# Function Definitions
# ---------------------

def perform_search(query, max_retries=1, backoff_factor=1):
    """Perform a DuckDuckGo search with retry and caching."""
    hashed_query = hashlib.md5(query.encode()).hexdigest()
    if hashed_query in search_cache:
        return search_cache[hashed_query]

    try_count = 0
    while try_count < max_retries:
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=2)
                domains = [{"link": result["href"]} for result in results]
                search_cache[hashed_query] = domains
                with open(CACHE_FILE, 'wb') as f:
                    pickle.dump(search_cache, f)
                return domains
        except HTTPError as e:
            if e.response.status_code == 429:
                try_count += 1
                time.sleep(backoff_factor * (2 ** try_count))
            else:
                break
        except Exception as e:
            try_count += 1
            time.sleep(backoff_factor * (2 ** try_count))

    return []

def verify_domain_with_llm(company_name, company_city, company_state, candidate_domains):
    """
    Use OpenAI's ChatCompletion to verify the most likely official domain.
    """
    if not candidate_domains:
        logging.info(f"No valid domains found for {company_name}. Skipping OpenAI verification.")
        return "No valid domains found"

    prompt = (
        f"Company: {company_name}\n"
        f"Location: {company_city}, {company_state}\n"
        f"Candidate domains: {', '.join(candidate_domains)}\n\n"
        "Which of these domains is the most likely official website for the company? "
        "Respond with just the domain name (e.g., https://example.com) and nothing else."
    )

    try:
        logging.info(f"Sending prompt to OpenAI for {company_name}...")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that identifies the correct official website for companies."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.2
        )
        # Extract the response content
        result = response["choices"][0]["message"]["content"].strip()
        logging.info(f"OpenAI Verification Result for {company_name}: {result}")

        # Extract only the domain using simple validation
        if result.startswith("http"):
            return result  # Return the domain as-is if it's valid
        else:
            logging.warning(f"Unexpected format for {company_name}: {result}")
            return "Invalid response from OpenAI"
    except Exception as e:
        logging.error(f"Error with OpenAI verification for {company_name}: {e}")
        return "Error verifying domain"


def process_companies(input_file, output_file, delay=1, limit=None):
    """Process companies from input CSV and save results."""
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        logging.error(f"Error reading input CSV: {e}")
        return

    if limit:
        df = df.head(limit)

    results = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        company_name = row.get("NAME", "").strip()
        city = row.get("CITY", "").strip()
        state = row.get("STATE", "").strip()

        if not company_name:
            continue

        query = f"{company_name} official website"
        search_results = perform_search(query)
        candidate_domains = [result["link"] for result in search_results]
        official_domain = verify_domain_with_llm(company_name, city, state, candidate_domains)

        results.append({
            "Company Name": company_name,
            "City": city,
            "State": state,
            "Official Domain": official_domain
        })

        time.sleep(delay)

    try:
        pd.DataFrame(results).to_csv(output_file, index=False)
    except Exception as e:
        logging.error(f"Error saving results: {e}")

if __name__ == "__main__":
    INPUT_FILE = r"C:\Users\DanielGeneralov\OneDrive - Tide Rock Holdings\Desktop\Domain_Finder.csv"
    OUTPUT_FILE = r"C:\Users\DanielGeneralov\OneDrive - Tide Rock Holdings\Desktop\Domain_Search_Results.csv"
    LIMIT = 5

    process_companies(INPUT_FILE, OUTPUT_FILE, delay=1, limit=LIMIT)

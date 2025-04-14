# import os
# import time
# import pandas as pd
# import requests
# from bs4 import BeautifulSoup
# from sentence_transformers import SentenceTransformer
# from supabase import create_client, Client
# from dotenv import load_dotenv
# import google.generativeai as genai
# import random

# # ----------------- Environment Setup -----------------
# load_dotenv()
# SUPABASE_URL = os.getenv("SUPABASE_URL")
# SUPABASE_KEY = os.getenv("SUPABASE_KEY")
# GEMINI_API_KEY = "AIzaSyAz1eU9YrwtPWg55sj_rJO0g_KoM5NHC58"

# # ----------------- Supabase & Gemini Config -----------------
# supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
# genai.configure(api_key=GEMINI_API_KEY)
# gemini = genai.GenerativeModel("gemini-1.5-pro")

# # ----------------- Embedding Model -----------------
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # ----------------- Constants -----------------
# CHUNK_SIZE = 500  # words
# TOP_K = 2  # similarity search top-k results

# # ----------------- Utility Functions -----------------

# def read_excel(file_path):
#     return pd.read_excel(file_path)

# def scrape_website(url):
#     try:
#         print(f"[INFO] Scraping: {url}")
#         response = requests.get(url, timeout=10)
#         soup = BeautifulSoup(response.text, "html.parser")
#         texts = [tag.get_text(strip=True) for tag in soup.find_all(["p", "h1", "h2", "h3", "li", "span"])]
#         return "\n".join(texts)
#     except Exception as e:
#         print(f"[ERROR] Failed to scrape {url}: {e}")
#         return ""

# def chunk_text(text, chunk_size=CHUNK_SIZE):
#     words = text.split()
#     return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# def embed_and_store(company, url, chunks):
#     for chunk in chunks:
#         embedding = model.encode(chunk).tolist()
#         response = supabase.table("company_embeddings").insert({
#             "company_name": company,
#             "url": url,
#             "content": chunk,
#             "embedding": embedding
#         }).execute()
#         if hasattr(response, "status_code") and response.status_code != 201:
#             print(f"[ERROR] Failed to insert into Supabase: {response}")

# def similarity_search(query, company):
#     query_vec = model.encode(query).tolist()
#     response = supabase.rpc("match_embeddings", {
#         "query_embedding": query_vec,
#         "match_count": TOP_K,
#         "company_name": company
#     }).execute()

#     if response.data:
#         return [item["content"] for item in response.data]
#     else:
#         print(f"[WARN] No match found for: {query}")
#         return []

# def ask_gemini(prompt):
#     try:
#         response = gemini.generate_content(prompt)
#         return response.text.strip()
#     except Exception as e:
#         return f"Gemini Error: {e}"

# def process_companies(df):
#     for idx, row in df.iterrows():
#         company = row['Company Name']
#         url = row['Company URL']
#         print(f"\n🚀 Processing company: {company}")
#         full_text = scrape_website(url)
#         if not full_text:
#             continue
#         chunks = chunk_text(full_text)
#         embed_and_store(company, url, chunks)
#         time.sleep(1)  # Respect web scraping etiquette

# def fill_columns(df):
#     columns_to_fill = [col for col in df.columns if col not in ['Company Name', 'Company URL']]
#     for idx, row in df.iterrows():
#         company = row["Company Name"]
#         print(f"\n🧠 Generating info for: {company}")

#         for col in columns_to_fill:
#             if pd.notna(row[col]):
#                 continue  # Skip if already filled
#             similar_chunks = similarity_search(col, company)
#             context = "\n".join(similar_chunks)

#             prompt = f"""
# You are given web content related to the company '{company}'.
# Using the context below, provide a **one-word or very short phrase** (maximum 3 words) that best answers the column titled: '{col}'.

# Context:
# {context}

# Your answer should be precise, no explanations, and strictly limited to 1 to 3 words.
# """


#             result = ask_gemini(prompt)
#             df.at[idx, col] = result
#             print(f"[{col}] => {result}")
#             time.sleep(random.uniform(3, 6))
#     return df

# # ----------------- Main -----------------
# if __name__ == "__main__":
#     df = read_excel("firstfive.xlsx")
#     process_companies(df)     # Step 1: Scrape and embed into Supabase
#     df = fill_columns(df)     # Step 2: Fill missing columns using Gemini AI
#     df.to_excel("filled_companies_info.xlsx", index=False)
#     print("\n✅ All company info enriched and saved to 'filled_companies_info.xlsx'")

import os
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from dotenv import load_dotenv
import openai
import random

# ----------------- Environment Setup -----------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ----------------- Supabase & OpenAI Config -----------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
openai.api_key = OPENAI_API_KEY

# ----------------- Embedding Model -----------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------- Constants -----------------
CHUNK_SIZE = 500  # words
TOP_K = 2  # similarity search top-k results

# ----------------- Utility Functions -----------------

def read_excel(file_path):
    return pd.read_excel(file_path)

def scrape_website(url):
    try:
        print(f"[INFO] Scraping: {url}")
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        texts = [tag.get_text(strip=True) for tag in soup.find_all(["p", "h1", "h2", "h3", "li", "span"])]
        return "\n".join(texts)
    except Exception as e:
        print(f"[ERROR] Failed to scrape {url}: {e}")
        return ""

def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_and_store(company, url, chunks):
    for chunk in chunks:
        embedding = model.encode(chunk).tolist()
        response = supabase.table("company_embeddings").insert({
            "company_name": company,
            "url": url,
            "content": chunk,
            "embedding": embedding
        }).execute()
        if hasattr(response, "status_code") and response.status_code != 201:
            print(f"[ERROR] Failed to insert into Supabase: {response}")

def similarity_search(query, company):
    query_vec = model.encode(query).tolist()
    response = supabase.rpc("match_embeddings", {
        "query_embedding": query_vec,
        "match_count": TOP_K,
        "company_name": company
    }).execute()

    if response.data:
        return [item["content"] for item in response.data]
    else:
        print(f"[WARN] No match found for: {query}")
        return []

def ask_openai(prompt):
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or use "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=30,
        )
        return completion.choices[0].message["content"].strip()
    except Exception as e:
        return f"OpenAI Error: {e}"

def process_companies(df):
    for idx, row in df.iterrows():
        company = row['Company Name']
        url = row['Company URL']
        print(f"\n🚀 Processing company: {company}")
        full_text = scrape_website(url)
        if not full_text:
            continue
        chunks = chunk_text(full_text)
        embed_and_store(company, url, chunks)
        time.sleep(1)  # Respect web scraping etiquette

def fill_columns(df):
    columns_to_fill = [col for col in df.columns if col not in ['Company Name', 'Company URL']]
    for idx, row in df.iterrows():
        company = row["Company Name"]
        print(f"\n🧠 Generating info for: {company}")

        for col in columns_to_fill:
            if pd.notna(row[col]):
                continue  # Skip if already filled
            similar_chunks = similarity_search(col, company)
            context = "\n".join(similar_chunks)

            prompt = f"""
You are given web content related to the company '{company}'.
Using the context below, provide a **one-word or very short phrase** (maximum 3 words) that best answers the column titled: '{col}'.

Context:
{context}

Your answer should be precise, no explanations, and strictly limited to 1 to 3 words.
"""

            result = ask_openai(prompt)
            df.at[idx, col] = result
            print(f"[{col}] => {result}")
            time.sleep(random.uniform(3, 6))
    return df

# ----------------- Main -----------------
if __name__ == "__main__":
    df = read_excel("firstfive.xlsx")
    process_companies(df)     # Step 1: Scrape and embed into Supabase
    df = fill_columns(df)     # Step 2: Fill missing columns using OpenAI
    df.to_excel("filled_companies_info.xlsx", index=False)
    print("\n✅ All company info enriched and saved to 'filled_companies_info.xlsx'")

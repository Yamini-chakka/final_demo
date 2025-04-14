import pandas as pd
import requests
 
# Load Excel file
df = pd.read_excel("names1.xlsx")
 
# SerpApi settings
API_KEY = '49a54d8eddd834567d9d74ebc34004b9d5630719b33a3d6c70203de4017b688e'
search_url = "https://serpapi.com/search"
 
# Function to get company URL
def get_company_url(company_name):
    params = {
        "engine": "google",
        "q": f"{company_name} official site",
        "api_key": API_KEY
    }
    try:
        response = requests.get(search_url, params=params)                                                                                                                                                                                                                          
        response.raise_for_status()  # Raise an error for HTTP issues
        data = response.json()
        return data['organic_results'][0]['link']
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return "URL Not Found"
    except (KeyError, IndexError):
        return "URL Not Found"
 
# Apply function
df['Company URL'] = df['Company Name'].apply(get_company_url)
 
# Save result
df.to_excel("companies_with_urls1.xlsx", index=False)
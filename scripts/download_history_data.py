import argparse
import requests
import json
from alpaca_config import API_KEY, API_SECRET

headers = {
    'APCA-API-KEY-ID': API_KEY,
    'APCA-API-SECRET-KEY': API_SECRET,
    'Content-Type': 'application/json'
}

params = {
    'symbols': 'BTC/USD',
    #'timeframe': '5Min',
    'timeframe': '1Hour',
    #'start': '2024-11-01',
    #'end': '2025-12-31',
    'start': '2025-07-01',
    'end': '2025-07-31',
    'limit':1000,
    'sort': 'asc'
}

api_url = 'https://data.alpaca.markets/v1beta3/crypto/us/bars'

def fetch_all_data(api_url, headers, params):
    all_data = []
    
    next_page_token = None
    while True:
        if next_page_token:
            params['page_token'] = next_page_token  # Add next_page_token for subsequent pages
        
        response = requests.get(api_url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"Error fetching data: {response.status_code} {response.text}")
            break
        
        data = response.json()
        
        # Append the data from this page to the all_data list
        #print(json.dumps(data, indent=2))
        if 'bars' in data and params['symbols'] in data['bars']: 
            all_data.extend(data['bars'][params['symbols']])
        
        # Check if there's a next page, if not, break the loop
        next_page_token = data.get('next_page_token')
        if not next_page_token:
            break  # No more pages, stop the loop
    
    return all_data

def save_to_file(data, filename="data.json"):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch BTC/USD bar data from Alpaca and save to JSON file."
    )
    parser.add_argument(
        "outfile",
        nargs="?",
        default="../data/input/data.json",
        help="Path to output JSON file (default: ../data/input/data.json)",
    )

    args = parser.parse_args()

    all_data = fetch_all_data(api_url, headers, params)
    save_to_file(all_data, args.outfile)

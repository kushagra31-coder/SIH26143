import os
import sys
import requests

def main():
    print("--- STEP 0.1: Copernicus Data Space Sentinel-1 Scene Check ---")
    
    username = os.environ.get("COPERNICUS_USER")
    password = os.environ.get("COPERNICUS_PASS")
    
    if not username or not password:
        print("FAIL: COPERNICUS_USER or COPERNICUS_PASS environment variables are not set.")
        print("Cannot authenticate to Copernicus Data Space to verify Sentinel-1 scene availability.")
        sys.exit(1)
        
    print("Copernicus credentials found. Attempting to get auth token...")
    
    # Get token
    token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    auth_data = {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": username,
        "password": password
    }
    
    try:
        response = requests.post(token_url, data=auth_data, headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=10)
        response.raise_for_status()
        access_token = response.json().get("access_token")
        print("Successfully obtained access token.")
    except Exception as e:
        print(f"FAIL: Authentication failed. Error: {e}")
        sys.exit(1)
        
    # Search for Wakashio scene: approx July 25 to Aug 15, 2020. Mauritius: Lat -20.4, Lon 57.7
    # Polygon around Mauritius
    mauritius_polygon = "POLYGON((57.2 -20.6, 58.0 -20.6, 58.0 -19.9, 57.2 -19.9, 57.2 -20.6))"
    search_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    
    # Query for Sentinel-1 GRD in the window
    query_params = {
        "$filter": f"Collection/Name eq 'SENTINEL-1' and OData.CSC.Intersects(footprint=geography'SRID=4326;{mauritius_polygon}') and ContentDate/Start gt 2020-07-20T00:00:00.000Z and ContentDate/Start lt 2020-08-15T00:00:00.000Z",
        "$top": 5
    }
    
    try:
        print(f"Querying catalog for Mauritius Sentinel-1 scenes (July/Aug 2020)...")
        search_resp = requests.get(search_url, params=query_params, timeout=10)
        search_resp.raise_for_status()
        
        results = search_resp.json().get("value", [])
        if not results:
            print("FAIL: No Sentinel-1 scenes found for the Wakashio incident area/time.")
            sys.exit(1)
            
        print(f"PASS: Found {len(results)} Sentinel-1 scenes.")
        print(f"Evidence - Top scene ID: {results[0].get('Id')}")
        print(f"Evidence - Top scene Name: {results[0].get('Name')}")
        sys.exit(0)
        
    except Exception as e:
        print(f"FAIL: Search query failed. Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

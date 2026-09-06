import os
import sys
import requests
import json
import time
from datetime import datetime, timedelta

def main():
    print("--- STEP 0.2: Global Fishing Watch (GFW) API Check ---")
    
    token = os.environ.get("GFW_API_TOKEN")
    if not token:
        print("FAIL: GFW_API_TOKEN environment variable is not set.")
        print("Cannot authenticate to GFW API.")
        sys.exit(1)
        
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Check 1: Identity Search
    print("\n[1] Checking Vessel Identity Search...")
    identity_url = "https://gateway.api.globalfishingwatch.org/v3/vessels/search"
    identity_params = {
        "query": "imo:9337119",
        "datasets[0]": "public-global-vessel-identity:latest"
    }
    
    try:
        resp = requests.get(identity_url, headers=headers, params=identity_params, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("entries", [])
        if results:
            vessel_id = None
            if results[0].get("combinedSourcesInfo"):
                vessel_id = results[0]["combinedSourcesInfo"][0].get("vesselId")
            if not vessel_id and results[0].get("registryInfo"):
                vessel_id = results[0]["registryInfo"][0].get("id")
            
            if not vessel_id:
                print("FAIL: Wakashio found but ID could not be extracted.")
                sys.exit(1)
            print(f"PASS: Found vessel Wakashio (GFW ID: {vessel_id})")
        else:
            print("FAIL: Wakashio not found in identity search.")
            sys.exit(1)
    except Exception as e:
        print(f"FAIL: Identity search failed. Error: {e}")
        sys.exit(1)
        
    # Check 2: Events
    print("\n[2] Checking Events Query...")
    events_url = f"https://gateway.api.globalfishingwatch.org/v3/events"
    events_params = {
        "vessels": vessel_id,
        "start-date": "2020-07-20",
        "end-date": "2020-08-01",
        "datasets[0]": "public-global-encounters-events:latest",
        "datasets[1]": "public-global-loitering-events-v1:latest",
        "datasets[2]": "public-global-port-visits-c2-events:latest"
    }
    try:
        events_resp = requests.get(events_url, headers=headers, params=events_params, timeout=10)
        if events_resp.status_code == 404:
            print("PASS: No fishing/loitering events found for Wakashio (expected for bulk carrier).")
        else:
            events_resp.raise_for_status()
            events = events_resp.json().get("entries", [])
            print(f"PASS: Found {len(events)} events for Wakashio in window.")
    except Exception as e:
        print(f"FAIL: Events query failed. Error: {e}")
        sys.exit(1)
        
    # Check 3: Presence/Tracks with 4wings Report Endpoint
    tracks_url = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"
    
    start_date = datetime(2020, 7, 20)
    end_date = datetime(2020, 8, 1)
    chunk_days = 3
    
    current_start = start_date
    all_records = []
    
    print("\n[3] Checking Presence Data (Trajectory/Resolution) with 4wings report endpoint...")
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=chunk_days), end_date)
        
        # 4Wings typically accepts datasets, date-range, and filters
        tracks_params = {
            "start-date": current_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end-date": current_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "datasets[0]": "public-global-ais-vessel-presence:v3.0",
            "vessels": vessel_id,
            "format": "json"
        }
        
        chunk_success = False
        for attempt in range(3):
            try:
                print(f"  Requesting {tracks_params['start-date']} to {tracks_params['end-date']} (Attempt {attempt+1}/3)...")
                tracks_resp = requests.get(tracks_url, headers=headers, params=tracks_params, timeout=20)
                
                if tracks_resp.status_code in [404, 422]:
                    print(f"  FAIL: Tracks query returned {tracks_resp.status_code}. Endpoint or dataset might be incorrect. Response: {tracks_resp.text}")
                    sys.exit(1)
                    
                tracks_resp.raise_for_status()
                track_data = tracks_resp.json()
                
                # 4wings report format is usually a list of records or a dictionary with an 'entries' array
                if isinstance(track_data, list):
                    all_records.extend(track_data)
                elif isinstance(track_data, dict) and "entries" in track_data:
                    all_records.extend(track_data["entries"])
                elif isinstance(track_data, dict) and "data" in track_data:
                    all_records.extend(track_data["data"])
                else:
                    all_records.append(track_data) # fallback
                
                chunk_success = True
                break  # Success, exit retry loop
            except Exception as e:
                print(f"  Timeout/Error on attempt {attempt+1}: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                
        if not chunk_success:
            print("FAIL: Tracks query failed after 3 retries due to timeouts.")
            sys.exit(1)
            
        current_start = current_end
        time.sleep(1) # Small delay between chunks

    print(f"PASS: Pulled track data. Found {len(all_records)} total records across all chunks.")
    if len(all_records) > 1:
        first_ts = all_records[0]
        second_ts = all_records[1]
        print(f"Evidence - Temporal Data check:")
        print(f"  Point 1: {str(first_ts)[:100]}")
        print(f"  Point 2: {str(second_ts)[:100]}")
    else:
        print(f"Evidence - Found {len(all_records)} records. Cannot determine temporal resolution. Data: {all_records}")

if __name__ == "__main__":
    main()

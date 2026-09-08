import os
import requests
import json

def test_events_api():
    token = os.environ.get("GFW_API_TOKEN")
    if not token:
        print("FAIL: No token.")
        return
        
    headers = {"Authorization": f"Bearer {token}"}
    
    # Tiny date range and the Mauritius bounding box
    url = "https://gateway.api.globalfishingwatch.org/v3/events"
    params = {
        "start-date": "2020-07-20T00:00:00Z",
        "end-date": "2020-07-22T00:00:00Z",
        "bbox": "56.0,-21.0,58.0,-19.0", # minLon,minLat,maxLon,maxLat
        "datasets[0]": "public-global-encounters-events:latest",
        "datasets[1]": "public-global-loitering-events-v1:latest",
        "datasets[2]": "public-global-port-visits-c2-events:latest"
    }
    
    print(f"Testing GFW Events API spatial query...")
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        print(f"Status Code: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            entries = data.get("entries", [])
            print(f"PASS: Events API works! Returned {len(entries)} events.")
        elif resp.status_code == 404:
            print("PASS (Empty): Events API works but found 0 events for this box/time.")
        else:
            print(f"FAIL: Events API failed with {resp.status_code}.")
            print(resp.text[:200])
            
    except Exception as e:
        print(f"FAIL: Error: {e}")

if __name__ == "__main__":
    test_events_api()

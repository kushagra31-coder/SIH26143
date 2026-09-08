import os, requests

token = os.environ.get("GFW_API_TOKEN")
headers = {"Authorization": f"Bearer {token}"}
vessel_id = '1854dc7b1-12d9-33f7-eed9-dccb56f7a860'
tracks_url = f'https://gateway.api.globalfishingwatch.org/v3/vessels/{vessel_id}/tracks'
tracks_params = {
    'start-date': '2020-07-20T00:00:00Z',
    'end-date': '2020-08-01T00:00:00Z',
    'datasets[0]': 'public-global-ais:latest'
}
resp = requests.get(tracks_url, headers=headers, params=tracks_params)
print('Tracks status:', resp.status_code)
if resp.status_code == 200:
    data = resp.json()
    if data:
        records = data[0].get('records', [])
        print(f'Found {len(records)} track points.')
        if len(records) >= 2:
            print(f'Point 1 TS: {records[0].get("timestamp")}')
            print(f'Point 2 TS: {records[1].get("timestamp")}')
